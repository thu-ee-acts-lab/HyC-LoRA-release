import math
import torch
import typing
import bitsandbytes as bnb
import torch.nn.functional as F
import bitsandbytes.functional as BF

from operators.layernorm_kernels import layernorm_forward, layernorm_backward
from operators.gelu_kernels import gelu_backward
from operators.softmax_kernels import softmax_backward

from ..utils.compute_utils import(
    hidden_to_head_shape, head_to_hidden_shape,
    lora_forward, lora_backward,
    repeat_kv, repeat_kv_backward
)

class FusedRobertaLayerFunc(torch.autograd.Function):
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
        head_dim: int
    ): 
        # compute q,k,v
        # forward process: q_proj
        q, q_main, q_lora_a = lora_forward(w_q, w_q_quant_state, w_q_lora_a, w_q_lora_b, b_q, x)
        
        # forward process: k_proj
        k, k_main, k_lora_a = lora_forward(w_k, w_k_quant_state, w_k_lora_a, w_k_lora_b, b_k, x)

        # forward process: v_proj
        v, v_main, v_lora_a = lora_forward(w_v, w_v_quant_state, w_v_lora_a, w_v_lora_b, b_v, x)
        
        # reshape
        q = hidden_to_head_shape(q, num_heads)
        k = hidden_to_head_shape(k, num_heads)
        v = hidden_to_head_shape(v, num_heads)
        
        ctx.q_shape = q.shape

        # q,k,v: [bsz, num_heads, q_len, head_dim]
        # notice forward process no need to drop heads
        bsz, num_heads, q_len, head_dim = q.shape

        # forward: S = Q @ K.T / sqrt(d_k)
        s = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
        # apply mask
        if attention_mask is not None:
            s = s + attention_mask

        # forward: softmax
        a = torch.softmax(s, dim=-1, dtype=v.dtype)  # [bsz, num_heads, q_len, q_len]
        
        # forward: O = A @ V
        o = a @ v
        
        # reshape
        o = head_to_hidden_shape(o)

        # forward process: o_proj
        o_final, o_final_main, o_final_lora_a = lora_forward(w_o, w_o_quant_state, w_o_lora_a, w_o_lora_b, b_o, o)
        
        # layernorm or rmsnorm (with residual connection)
        o_final += x
        x_medium, mean_1, rstd_1, _, _ = layernorm_forward(o_final, norm_weight_1, norm_bias_1, eps = 1e-5)

        # forward process: up_proj
        up, up_main, up_lora_a = lora_forward(w_up, w_up_quant_state, w_up_lora_a, w_up_lora_b, b_up, x_medium)
        
        # activation function
        fn = torch.nn.functional.gelu(up)

        # forward process: down_proj
        down, down_main, down_lora_a = lora_forward(w_down, w_down_quant_state, w_down_lora_a, w_down_lora_b, b_down, fn)
        down += x_medium
        x_out, mean_2, rstd_2, block_size, num_warps = layernorm_forward(down, norm_weight_2, norm_bias_2, eps = 1e-5)
        
        ctx.save_for_backward(
            ### activations (attention) ###
            x, # buffer for q, k, v
            q_lora_a, # buffer for lora (qkv)
            k_lora_a, # buffer for lora (qkv)
            v_lora_a, # buffer for lora (qkv)
            q, # buffer for attn
            k, # buffer for attn
            v, # buffer for attn
            a, # buffer for attn
            o, # buffer for o
            o_final_lora_a, # buffer for lora (o)
            o_final, # buffer for layernorm
            mean_1, # buffer for layernorm
            rstd_1, # buffer for layernorm
            ### activations (mlp) ###
            x_medium, # buffer for up
            up_lora_a, # buffer for lora (up)
            up, # buffer for gelu/silu
            fn, # buffer for down
            down_lora_a, # buffer for lora (down)
            down, # buffer for layernorm
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

        return x_out
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            w_q_quant_state,
            w_k_quant_state,
            w_v_quant_state,
            w_o_quant_state,
            w_up_quant_state,
            w_down_quant_state,
        ) = ctx.quant_state
        
        (
            x, # buffer for q, k, v
            q_lora_a, # buffer for lora (qkv)
            k_lora_a, # buffer for lora (qkv)
            v_lora_a, # buffer for lora (qkv)
            q, # buffer for attn
            k, # buffer for attn
            v, # buffer for attn
            a, # buffer for attn
            o, # buffer for o
            o_final_lora_a, # buffer for lora (o)
            o_final, # buffer for layernorm
            mean_1, # buffer for layernorm
            rstd_1, # buffer for layernorm
            ### activations (mlp) ###
            x_medium, # buffer for up
            up_lora_a, # buffer for lora (up)
            up, # buffer for gelu/silu
            fn, # buffer for down
            down_lora_a, # buffer for lora (down)
            down, # buffer for layernorm
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
        grad_layernorm_2, _, _ = layernorm_backward(
            grad_output, down, norm_weight_2, norm_bias_2, mean_2, rstd_2, # TODO: other params
            True, 1e-5, ctx.num_warps, ctx.block_size
        )
        
        # down proj part
        grad_w_down_lora_a, grad_w_down_lora_b, grad_down = lora_backward(w_down, w_down_quant_state, w_down_lora_a, w_down_lora_b, fn, down_lora_a, grad_layernorm_2)
        
        # TODO: activation backward
        # activation part
        grad_fn = gelu_backward(up, grad_down)
        
        # up proj part
        grad_w_up_lora_a, grad_w_up_lora_b, grad_up = lora_backward(w_up, w_up_quant_state, w_up_lora_a, w_up_lora_b, x_medium, up_lora_a, grad_fn)
        
        # residual connection
        grad_medium = grad_layernorm_2 + grad_up
        
        # layernorm
        grad_x_layernorm_1, _, _ = layernorm_backward(
            grad_medium, o_final, norm_weight_1, norm_bias_1, mean_1, rstd_1, # TODO: other params
            True, 1e-5, ctx.num_warps, ctx.block_size
        )
        
        # o part
        grad_w_o_lora_a, grad_w_o_lora_b, grad_o = lora_backward(w_o, w_o_quant_state, w_o_lora_a, w_o_lora_b, o, o_final_lora_a, grad_x_layernorm_1)
        
        # reshape
        grad_o = hidden_to_head_shape(grad_o, ctx.num_heads)

        # backward of second GEMM: O = A @ V
        # d L / d V = A.T @ d L / d O
        grad_v = a.transpose(-2, -1) @ grad_o
        grad_a = grad_o @ v.transpose(-2, -1)

        # backward of softmax
        grad_s = softmax_backward(a, grad_a)

        # backward of first GEMM: S = Q @ K.T / sqrt(d_k)
        grad_s = grad_s / math.sqrt(ctx.head_dim)
        # d L / d K = (d L / d S)^T @ Q
        grad_k = grad_s.transpose(-2, -1) @ q
        # d L / d Q = d L / d S @ K
        grad_q = grad_s @ k

        grad_q = head_to_hidden_shape(grad_q)
        grad_k = head_to_hidden_shape(grad_k)
        grad_v = head_to_hidden_shape(grad_v)
        
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
        ) + (None,) * 3


class FusedRobertaLayer(torch.nn.Module):
    def __init__(
        self,
    ):
        super(FusedRobertaLayer, self).__init__()


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
        x_out = FusedRobertaLayerFunc.apply(
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
        )
        
        return x_out