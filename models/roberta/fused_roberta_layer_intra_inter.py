import math
import torch
import typing
import bitsandbytes as bnb

from operators.layernorm_kernels import layernorm_forward, layernorm_backward
from operators.gelu_kernels import gelu_backward
from operators.softmax_kernels import softmax_backward

from operators.compress_function import (
    compression_pack_channel_base,
    compression_pack_quant_base,
    compression_pack_softmax_base,
    outlier_addition_fuse_decompression_dequantization,
    decompression_dequantization,
    update_dict,
)

from ..utils.compute_utils import(
    hidden_to_head_shape, head_to_hidden_shape,
    lora_forward, lora_backward,
    repeat_kv, repeat_kv_backward
)

class FusedRobertaLayerIntraInterFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        #############attention part#############
        w_q: torch.Tensor,
        b_q: torch.Tensor,
        w_q_quant_state: typing.Tuple,
        w_q_lora_a: torch.Tensor,
        w_q_lora_b: torch.Tensor,
        ####################################
        w_k: torch.Tensor,
        b_k: torch.Tensor,
        w_k_quant_state: typing.Tuple,
        w_k_lora_a: torch.Tensor,
        w_k_lora_b: torch.Tensor,
        ####################################
        w_v: torch.Tensor,
        b_v: torch.Tensor,
        w_v_quant_state: typing.Tuple,
        w_v_lora_a: torch.Tensor,
        w_v_lora_b: torch.Tensor,
        ####################################
        w_o: torch.Tensor,
        b_o: torch.Tensor,
        w_o_quant_state: typing.Tuple,
        w_o_lora_a: torch.Tensor,
        w_o_lora_b: torch.Tensor,
        ####################################
        norm_weight_1: torch.Tensor,
        norm_bias_1: torch.Tensor,
        #############mlp part#############
        w_up: torch.Tensor,
        b_up: torch.Tensor,
        w_up_quant_state: typing.Tuple,
        w_up_lora_a: torch.Tensor,
        w_up_lora_b: torch.Tensor,
        ####################################
        w_down: torch.Tensor,
        b_down: torch.Tensor,
        w_down_quant_state: typing.Tuple,
        w_down_lora_a: torch.Tensor,
        w_down_lora_b: torch.Tensor,
        ####################################
        norm_weight_2: torch.Tensor,
        norm_bias_2: torch.Tensor,
        ###############other################
        attention_mask: torch.Tensor,
        num_heads: int,
        head_dim: int,
        ###############about statistics################
        iteration: int,
        iteration_threshold: int,
        static_value: dict,
        softmax_outlier_ratio: float,
        layernorm_outlier_ratio: float,
        q_bit: int,
    ): 
        # compute q,k,v
        # forward process: q_proj
        q, q_main, q_lora_a = lora_forward(w_q, w_q_quant_state, w_q_lora_a, w_q_lora_b, b_q, x)
        
        # forward process: k_proj
        k, k_main, k_lora_a = lora_forward(w_k, w_k_quant_state, w_k_lora_a, w_k_lora_b, b_k, x)

        # forward process: v_proj
        v, v_main, v_lora_a = lora_forward(w_v, w_v_quant_state, w_v_lora_a, w_v_lora_b, b_v, x)
        
        #* compress x
        x_q, x_scale = compression_pack_quant_base(
            x=x, q_bit=q_bit, 
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['x']
        )
        
        #* compress q_main, k_main, v_main
        q_main_q, q_main_scale = compression_pack_quant_base(
            x=q_main, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['q']
        )
        k_main_q, k_main_scale = compression_pack_quant_base(
            x=k_main, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['k']
        )
        v_main_q, v_main_scale = compression_pack_quant_base(
            x=v_main, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['v']
        )
        del q_main, k_main, v_main
        
        # reshape
        q = hidden_to_head_shape(q, num_heads)
        k = hidden_to_head_shape(k, num_heads)
        v = hidden_to_head_shape(v, num_heads)
        
        ctx.q_shape = q.shape

        # forward: S = Q @ K.T / sqrt(d_k)
        s = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
        del q, k
        
        # apply mask
        if attention_mask is not None:
            s = s + attention_mask

        # forward: softmax
        a = torch.softmax(s, dim=-1, dtype=v.dtype)  # [bsz, num_heads, q_len, q_len]
        
        # forward: O = A @ V
        o = a @ v
        
        #* compress a
        a_o, a_threshold = compression_pack_softmax_base(
            x=a, o_ratio=softmax_outlier_ratio, it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['a']
        )
        ctx.a_shape = a.shape
        del a, v
        
        # reshape
        o = head_to_hidden_shape(o)

        # forward process: o_proj
        o_final, _, o_final_lora_a = lora_forward(w_o, w_o_quant_state, w_o_lora_a, w_o_lora_b, b_o, o)
        
        #* compress o
        o_q, o_scale = compression_pack_quant_base(
            x=o, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['o']
        )
        del o
        
        # layernorm or rmsnorm (with residual connection)
        o_final += x
        x_medium, mean_1, rstd_1, _, _ = layernorm_forward(o_final, norm_weight_1, norm_bias_1, eps = 1e-5)
        
        o_final_o, o_final_q, o_final_channel_idx, o_final_scale = compression_pack_channel_base(
            x=o_final, o_ratio=layernorm_outlier_ratio, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['o_final']
        )

        # forward process: up_proj
        up, up_main, up_lora_a = lora_forward(w_up, w_up_quant_state, w_up_lora_a, w_up_lora_b, b_up, x_medium)
        
        x_medium_q, x_medium_scale = compression_pack_quant_base(
            x=x_medium, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['x_medium']
        )
        up_main_q, up_main_scale = compression_pack_quant_base(
            x=up_main, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['up']
        )
        del up_main
        
        # activation function
        fn = torch.nn.functional.gelu(up)

        # forward process: down_proj
        down, _, down_lora_a = lora_forward(w_down, w_down_quant_state, w_down_lora_a, w_down_lora_b, b_down, fn)
        down += x_medium
        
        x_out, mean_2, rstd_2, block_size, num_warps = layernorm_forward(down, norm_weight_2, norm_bias_2, eps = 1e-5)
        
        down_o, down_q, down_channel_idx, down_scale = compression_pack_channel_base(
            x=down, o_ratio=layernorm_outlier_ratio, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['down']
        )
        del down
        
        ctx.save_for_backward(
            ### activations (attention) ###
            x_q, x_scale, # buffer for q, k, v
            q_lora_a, # buffer for lora (qkv)
            k_lora_a, # buffer for lora (qkv)
            v_lora_a, # buffer for lora (qkv)
            q_main_q, q_main_scale, # q
            k_main_q, k_main_scale, # k
            v_main_q, v_main_scale, # v
            a_o, a_threshold, # a
            o_q, o_scale, # o
            o_final_lora_a, # buffer for lora (o)
            o_final_o, o_final_q, o_final_scale, # buffer for layernorm
            mean_1, # buffer for layernorm
            rstd_1, # buffer for layernorm
            ### activations (mlp) ###
            x_medium_q, x_medium_scale, # buffer for up
            up_lora_a, # buffer for lora (up)
            up_main_q, up_main_scale, # up
            down_lora_a, # buffer for lora (down)
            down_o, down_q, down_scale, # buffer for layernorm
            mean_2, # buffer for layernorm
            rstd_2, # buffer for layernorm
            ### weights (attention) ###
            w_q,
            b_q,
            w_q_lora_a,
            w_q_lora_b,
            #**********************
            w_k,
            b_k,
            w_k_lora_a,
            w_k_lora_b,
            #**********************
            w_v,
            b_v,
            w_v_lora_a,
            w_v_lora_b,
            #**********************
            w_o,
            b_o,
            w_o_lora_a,
            w_o_lora_b,
            #**********************
            norm_weight_1, 
            norm_bias_1,
            ### weights (mlp) ###
            #**********************
            w_up,
            b_up,
            w_up_lora_a,
            w_up_lora_b,
            #**********************
            w_down,
            b_down,
            w_down_lora_a,
            w_down_lora_b,
            #**********************
            norm_weight_2,
            norm_bias_2,
        )
        ctx.quant_state = (
            w_q_quant_state,
            w_k_quant_state,
            w_v_quant_state,
            w_o_quant_state,
            w_up_quant_state,
            w_down_quant_state,
        )
        ctx.num_heads = num_heads
        ctx.block_size = block_size
        ctx.num_warps = num_warps
        ctx.head_dim = head_dim
        ctx.q_bit = q_bit
        ctx.o_final_channel = o_final_channel_idx
        ctx.down_channel = down_channel_idx

        return x_out, x_scale, \
            q_main_scale, k_main_scale, v_main_scale, \
            a_threshold, o_scale, \
            o_final_channel_idx, o_final_scale, \
            x_medium_scale, \
            up_main_scale, down_channel_idx, down_scale
            
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, *args):
        (
            w_q_quant_state,
            w_k_quant_state,
            w_v_quant_state,
            w_o_quant_state,
            w_up_quant_state,
            w_down_quant_state,
        ) = ctx.quant_state
        
        (
            ### activations (attention) ###
            x_q, x_scale, # buffer for q, k, v
            q_lora_a, # buffer for lora (qkv)
            k_lora_a, # buffer for lora (qkv)
            v_lora_a, # buffer for lora (qkv)
            q_main_q, q_main_scale, # q
            k_main_q, k_main_scale, # k
            v_main_q, v_main_scale, # v
            a_o, a_threshold, # a
            o_q, o_scale, # o
            o_final_lora_a, # buffer for lora (o)
            o_final_o, o_final_q, o_final_scale, # buffer for layernorm
            mean_1, # buffer for layernorm
            rstd_1, # buffer for layernorm
            ### activations (mlp) ###
            x_medium_q, x_medium_scale, # buffer for up
            up_lora_a, # buffer for lora (up)
            up_main_q, up_main_scale, # up
            down_lora_a, # buffer for lora (down)
            down_o, down_q, down_scale, # buffer for layernorm
            mean_2, # buffer for layernorm
            rstd_2, # buffer for layernorm
            ### weights (attention) ###
            w_q,
            b_q,
            w_q_lora_a,
            w_q_lora_b,
            #**********************
            w_k,
            b_k,
            w_k_lora_a,
            w_k_lora_b,
            #**********************
            w_v,
            b_v,
            w_v_lora_a,
            w_v_lora_b,
            #**********************
            w_o,
            b_o,
            w_o_lora_a,
            w_o_lora_b,
            #**********************
            norm_weight_1, 
            norm_bias_1,
            ### weights (mlp) ###
            #**********************
            w_up,
            b_up,
            w_up_lora_a,
            w_up_lora_b,
            #**********************
            w_down,
            b_down,
            w_down_lora_a,
            w_down_lora_b,
            #**********************
            norm_weight_2,
            norm_bias_2,
        ) = ctx.saved_tensors
        
        grad_output = grad_output.to(torch.bfloat16)
        
        # layernorm
        down = outlier_addition_fuse_decompression_dequantization(down_q, down_scale, down_o, ctx.down_channel, ctx.q_bit)
        grad_layernorm_2, _, _ = layernorm_backward(
            grad_output, down, norm_weight_2, norm_bias_2, mean_2, rstd_2, # TODO: other params
            True, 1e-5, ctx.num_warps, ctx.block_size
        )
        
        #* dequantize up_main, and recompute grad_up
        up_main = decompression_dequantization(up_main_q, up_main_scale, ctx.q_bit)
        del up_main_q, up_main_scale
        up = up_main + up_lora_a.to(up_main.dtype) @ w_up_lora_b.to(up_main.dtype)
        
        # down proj part
        fn = torch.nn.functional.gelu(up)
        grad_w_down_lora_a, grad_w_down_lora_b, grad_down = lora_backward(w_down, w_down_quant_state, w_down_lora_a, w_down_lora_b, fn, down_lora_a, grad_layernorm_2)
        
        # activation part
        grad_fn = gelu_backward(up, grad_down)
        
        # up proj part
        x_medium = decompression_dequantization(x_medium_q, x_medium_scale, ctx.q_bit)
        grad_w_up_lora_a, grad_w_up_lora_b, grad_up = lora_backward(w_up, w_up_quant_state, w_up_lora_a, w_up_lora_b, x_medium, up_lora_a, grad_fn)
        
        # residual connection
        grad_medium = grad_layernorm_2 + grad_up
        
        # layernorm
        o_final = outlier_addition_fuse_decompression_dequantization(o_final_q, o_final_scale, o_final_o, ctx.o_final_channel, ctx.q_bit)
        grad_x_layernorm_1, _, _ = layernorm_backward(
            grad_medium, o_final, norm_weight_1, norm_bias_1, mean_1, rstd_1, # TODO: other params
            True, 1e-5, ctx.num_warps, ctx.block_size
        )
        
        # o part
        o = decompression_dequantization(o_q, o_scale, ctx.q_bit)
        grad_w_o_lora_a, grad_w_o_lora_b, grad_o = lora_backward(w_o, w_o_quant_state, w_o_lora_a, w_o_lora_b, o, o_final_lora_a, grad_x_layernorm_1)
        
        # reshape
        grad_o = hidden_to_head_shape(grad_o, ctx.num_heads)

        # backward of second GEMM: O = A @ V
        # d L / d V = A.T @ d L / d O
        a = a_o.to_dense()
        a = a.reshape(ctx.a_shape)
        v_main = decompression_dequantization(v_main_q, v_main_scale, ctx.q_bit)
        v = v_main + v_lora_a.to(v_main.dtype) @ w_v_lora_b.to(v_main.dtype)
        v = hidden_to_head_shape(v, ctx.num_heads)
        grad_v = a.transpose(-2, -1) @ grad_o
        grad_a = grad_o @ v.transpose(-2, -1)
        del a_o, v_main_q, v_main_scale, v_main, v

        # backward of softmax
        grad_s = softmax_backward(a, grad_a)
        del a, grad_a

        # backward of first GEMM: S = Q @ K.T / sqrt(d_k)
        q_main = decompression_dequantization(q_main_q, q_main_scale, ctx.q_bit)
        del q_main_q, q_main_scale
        k_main = decompression_dequantization(k_main_q, k_main_scale, ctx.q_bit)
        del k_main_q, k_main_scale
        q = q_main + q_lora_a.to(q_main.dtype) @ w_q_lora_b.to(q_main.dtype)
        q = hidden_to_head_shape(q, ctx.num_heads)
        del q_main
        k = k_main + k_lora_a.to(k_main.dtype) @ w_k_lora_b.to(k_main.dtype)
        k = hidden_to_head_shape(k, ctx.num_heads)
        del k_main
        
        grad_s = grad_s / math.sqrt(ctx.head_dim)
        # d L / d K = (d L / d S)^T @ Q
        grad_k = grad_s.transpose(-2, -1) @ q
        # d L / d Q = d L / d S @ K
        grad_q = grad_s @ k

        grad_q = head_to_hidden_shape(grad_q)
        grad_k = head_to_hidden_shape(grad_k)
        grad_v = head_to_hidden_shape(grad_v)
        
        x = decompression_dequantization(x_q, x_scale, ctx.q_bit)
        # backward of q_proj
        grad_w_q_lora_a, grad_w_q_lora_b, grad_x = lora_backward(w_q, w_q_quant_state, w_q_lora_a, w_q_lora_b, x, q_lora_a, grad_q)

        # backward of k_proj
        grad_w_k_lora_a, grad_w_k_lora_b, grad_x_temp = lora_backward(w_k, w_k_quant_state, w_k_lora_a, w_k_lora_b, x, k_lora_a, grad_k)
        grad_x += grad_x_temp

        # backward of v_proj
        grad_w_v_lora_a, grad_w_v_lora_b, grad_x_temp = lora_backward(w_v, w_v_quant_state, w_v_lora_a, w_v_lora_b, x, v_lora_a, grad_v)
        grad_x += grad_x_temp
          
        # residual connection
        grad_input = grad_x + grad_x_layernorm_1
        
        return (
            grad_input,
            #############attention part#############
            None,
            None,
            None,
            grad_w_q_lora_a,
            grad_w_q_lora_b,
            ####################################
            None,
            None,
            None,
            grad_w_k_lora_a,
            grad_w_k_lora_b,
            ####################################
            None,
            None,
            None,
            grad_w_v_lora_a,
            grad_w_v_lora_b,
            ####################################
            None,
            None,
            None,
            grad_w_o_lora_a,
            grad_w_o_lora_b,
            ####################################
            None,
            None,
            ####################################
            None,
            None,
            None,
            grad_w_up_lora_a,
            grad_w_up_lora_b,
            ####################################
            None,
            None,
            None,
            grad_w_down_lora_a,
            grad_w_down_lora_b,
            ####################################
            None,
            None,
        ) + (None,) * 9


class FusedRobertaLayerIntraInter(torch.nn.Module):
    
    def __init__(
        self,
    ):
        super(FusedRobertaLayerIntraInter, self).__init__()
        
        self.iteration = 0
        self.iteration_threshold = 5
        self.softmax_outlier_ratio = 0.05
        self.layernorm_outlier_ratio = 0.005
        self.q_bit = 2

        self.static_value = {
            'x': {'scale': None},
            'q': {'scale': None},
            'k': {'scale': None},
            'v': {'scale': None},
            'a': {'outlier': None},
            'o': {'scale': None},
            'o_final': {'outlier_channel_index': None, 'scale': None},
            'x_medium': {'scale': None},
            'up': {'scale': None},
            'down': {'outlier_channel_index': None, 'scale': None},
        }

  
    def set_hyclora_config(self, hyclora_config):
        self.hyclora_config = hyclora_config
        self.use_hyclora = hyclora_config.use_hyclora
        self.iteration_threshold = hyclora_config.iteration_threshold
        self.softmax_outlier_ratio = hyclora_config.softmax_outlier_ratio
        self.layernorm_outlier_ratio = hyclora_config.layernorm_outlier_ratio
        self.q_bit = hyclora_config.q_bit


    def forward(
        self,
        input: torch.Tensor,
        ############################################
        q_proj_base: bnb.nn.modules.Linear4bit,
        q_proj_lora_a: torch.nn.Linear,
        q_proj_lora_b: torch.nn.Linear,
        k_proj_base: bnb.nn.modules.Linear4bit,
        k_proj_lora_a: torch.nn.Linear,
        k_proj_lora_b: torch.nn.Linear,
        v_proj_base: bnb.nn.modules.Linear4bit,
        v_proj_lora_a: torch.nn.Linear,
        v_proj_lora_b: torch.nn.Linear,
        o_proj_base: bnb.nn.modules.Linear4bit,
        o_proj_lora_a: torch.nn.Linear,
        o_proj_lora_b: torch.nn.Linear,
        norm_weight_1: torch.Tensor,
        norm_bias_1: torch.Tensor,
        ############################################
        up_proj_base: bnb.nn.modules.Linear4bit,
        up_proj_lora_a: torch.nn.Linear,
        up_proj_lora_b: torch.nn.Linear,
        down_proj_base: bnb.nn.modules.Linear4bit,
        down_proj_lora_a: torch.nn.Linear,
        down_proj_lora_b: torch.nn.Linear,
        norm_weight_2: torch.Tensor,
        norm_bias_2: torch.Tensor,
        ############################################
        attention_mask: torch.Tensor,
        num_heads: int,
        head_dim: int,
        ############################################
    ):
        x_out, x_scale, \
        q_scale, k_scale, v_scale, \
        a_threshold, o_scale, \
        o_final_channel_idx, o_final_scale, \
        x_medium_scale, \
        up_scale, \
        down_channel_idx, down_scale = FusedRobertaLayerIntraInterFunc.apply(
            input,
            #############attention part#############
            q_proj_base.weight,
            q_proj_base.bias,
            q_proj_base.weight.quant_state,
            q_proj_lora_a.default.weight.T,
            q_proj_lora_b.default.weight.T,
            ####################################
            k_proj_base.weight,
            k_proj_base.bias,
            k_proj_base.weight.quant_state,
            k_proj_lora_a.default.weight.T,
            k_proj_lora_b.default.weight.T,
            ####################################
            v_proj_base.weight,
            v_proj_base.bias,
            v_proj_base.weight.quant_state,
            v_proj_lora_a.default.weight.T,
            v_proj_lora_b.default.weight.T,
            ####################################
            o_proj_base.weight,
            o_proj_base.bias,
            o_proj_base.weight.quant_state,
            o_proj_lora_a.default.weight.T,
            o_proj_lora_b.default.weight.T,
            ####################################
            norm_weight_1,
            norm_bias_1,
            #############mlp part#############
            up_proj_base.weight,
            up_proj_base.bias,
            up_proj_base.weight.quant_state,
            up_proj_lora_a.default.weight.T,
            up_proj_lora_b.default.weight.T,
            ####################################
            down_proj_base.weight,
            down_proj_base.bias,
            down_proj_base.weight.quant_state,
            down_proj_lora_a.default.weight.T,
            down_proj_lora_b.default.weight.T,
            ####################################
            norm_weight_2,
            norm_bias_2,
            ####################################
            attention_mask,
            num_heads,
            head_dim,
            ####################################
            self.iteration,
            self.iteration_threshold,
            self.static_value,
            self.softmax_outlier_ratio,
            self.layernorm_outlier_ratio,
            self.q_bit,
        )
        
        if self.iteration < self.iteration_threshold:
            self.static_value['x'] = update_dict(self.static_value['x'], {'scale': x_scale}, self.iteration)
            self.static_value['q'] = update_dict(self.static_value['q'], {'scale': q_scale}, self.iteration)
            self.static_value['k'] = update_dict(self.static_value['k'], {'scale': k_scale}, self.iteration)
            self.static_value['v'] = update_dict(self.static_value['v'], {'scale': v_scale}, self.iteration)
            self.static_value['a'] = update_dict(self.static_value['a'], {'outlier': a_threshold}, self.iteration)
            self.static_value['o'] = update_dict(self.static_value['o'], {'scale': o_scale}, self.iteration)
            self.static_value['o_final'] = update_dict(self.static_value['o_final'], {'outlier_channel_index': o_final_channel_idx, 'scale': o_final_scale}, self.iteration)
            self.static_value['x_medium'] = update_dict(self.static_value['x_medium'], {'scale': x_medium_scale}, self.iteration)
            self.static_value['up'] = update_dict(self.static_value['up'], {'scale': up_scale}, self.iteration)
            self.static_value['down'] = update_dict(self.static_value['down'], {'outlier_channel_index': down_channel_idx, 'scale': down_scale}, self.iteration)
        self.iteration += 1
        
        return x_out