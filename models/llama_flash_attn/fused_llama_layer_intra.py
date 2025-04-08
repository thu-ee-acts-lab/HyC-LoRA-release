import math
import torch
import typing
import bitsandbytes as bnb
import torch.nn.functional as F
import bitsandbytes.functional as BF

from operators.rope_kernels import rope_forward, rope_backward, calculate_settings
from operators.silu_kernels import silu_backward
from operators.rmsnorm_updatable_kernels import rmsnorm_backward, rmsnorm_forward
from operators.compress_function import (
    compression_pack_channel_base,
    compression_pack_quant_base,
    compression_pack_quant_zp_base,
    outlier_addition_fuse_decompression_dequantization,
    decompression_dequantization,
    decompression_dequantization_with_zero_point,
    update_dict,
)

from ..utils.compute_utils import(
    hidden_to_head_shape, head_to_hidden_shape,
    lora_forward, lora_backward,
    unpad_input, pad_input
)

from einops import rearrange
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward, _flash_attn_varlen_backward


class FusedLlamaLayerIntraFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        #############attention part#############
        norm_weight_1: torch.Tensor,
        norm_bias_1: torch.Tensor,
        ####################################
        cos: torch.Tensor,
        sin: torch.Tensor,
        ####################################
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
        #############mlp part#############
        norm_weight_2: torch.Tensor,
        norm_bias_2: torch.Tensor,
        ####################################
        w_gate: torch.Tensor,
        b_gate: torch.Tensor,
        w_gate_quant_state: typing.Tuple,
        w_gate_lora_a: torch.Tensor,
        w_gate_lora_b: torch.Tensor,
        ####################################
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
        ###############other################
        attention_mask: torch.Tensor,
        num_heads: int,
        head_dim: int,
        ###############about statistics################
        iteration: int,
        iteration_threshold: int,
        static_value: dict,
        layernorm_outlier_ratio: float,
        q_bit: int,
    ):
        # layernorm or rmsnorm
        bsz, q_len, _ = x.size()
        x_norm_1, mean_1, rstd_1, _, _ = rmsnorm_forward(x, norm_weight_1, eps = 1e-5)
        
        #* compress the (copy of) x
        x_copy = x.clone()
        x_o, x_q, x_channel_idx, x_scale = compression_pack_channel_base(
            x=x_copy, o_ratio=layernorm_outlier_ratio, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['x']
        )

        # compute q,k,v
        # forward process: q_proj
        q, _, q_lora_a = lora_forward(w_q, w_q_quant_state, w_q_lora_a, w_q_lora_b, b_q, x_norm_1)

        # forward process: k_proj
        k, _, k_lora_a = lora_forward(w_k, w_k_quant_state, w_k_lora_a, w_k_lora_b, b_k, x_norm_1)

        # forward process: v_proj
        v, _, v_lora_a = lora_forward(w_v, w_v_quant_state, w_v_lora_a, w_v_lora_b, b_v, x_norm_1)
        
        #* compress x_norm_1
        x_norm_1_q, x_norm_1_scale = compression_pack_quant_base(
            x=x_norm_1, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['x_norm_1']
        )
        del x_norm_1
        
        # reshape
        q = hidden_to_head_shape(q, num_heads)
        k = hidden_to_head_shape(k, num_heads)
        v = hidden_to_head_shape(v, num_heads)
        
        ctx.q_shape = q.shape

        q = rope_forward(q.transpose(1, 2), cos, sin).transpose(1, 2)
        k = rope_forward(k.transpose(1, 2), cos, sin).transpose(1, 2)
        
        #! the q/k value must use zero-point to quantize, see https://arxiv.org/abs/2410.02367
        q_q, q_scale, q_zero_point = compression_pack_quant_zp_base(
            x=q, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['q']
        )
        k_q, k_scale, k_zero_point = compression_pack_quant_zp_base(
            x=k, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['k']
        )
        v_q, v_scale = compression_pack_quant_base(
            x=v, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['v']
        )

        #! flash_attn begin
        qkv = torch.stack([q, k, v], dim=2)
        qkv = qkv.transpose(1, 3)
        
        key_padding_mask = attention_mask
        nheads = qkv.shape[-2]
        qkv = rearrange(qkv, "b s three h d -> b s (three h d)")
        # print the memory usage
        x_unpad, indices, cu_q_lens, max_s = unpad_input(qkv, key_padding_mask)
        x_unpad = rearrange(
            x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads
        )
        
        softmax_scale = x_unpad.shape[-1] ** -0.5
        o_unpad, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward(
            q=x_unpad[:, 0], k=x_unpad[:, 1], v=x_unpad[:, 2], 
            cu_seqlens_q = cu_q_lens, cu_seqlens_k = cu_q_lens,
            max_seqlen_q=max_s, max_seqlen_k=max_s,
            dropout_p=0.0, softmax_scale=softmax_scale,
            causal=True, window_size=(-1, -1),
            alibi_slopes=None, return_softmax=False, block_table=None
        )
        
        o = rearrange(pad_input(rearrange(o_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len), "b s (h d) -> b s h d", h=nheads)
        o = o.view(bsz, q_len, num_heads * head_dim)
        #! flash_attn end

        # forward process: o_proj
        o_final, _, o_final_lora_a = lora_forward(w_o, w_o_quant_state, w_o_lora_a, w_o_lora_b, b_o, o)
        
        #* compress o
        o_q, o_scale = compression_pack_quant_base(
            x=o, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['o']
        )
        del o
        
        # residual connection
        x_medium = x + o_final
        del x, o_final
        
        # layernorm or rmsnorm
        x_norm_2, mean_2, rstd_2, block_size, num_warps = rmsnorm_forward(x_medium, norm_weight_2, eps = 1e-5)
        
        #* compress the (copy of) x_medium
        x_medium_copy = x_medium.clone()
        x_medium_o, x_medium_q, x_medium_channel_idx, x_medium_scale = compression_pack_channel_base(
            x=x_medium_copy, o_ratio=layernorm_outlier_ratio, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['x_medium']
        )
        
        # forward process: gate_proj
        gate, _, gate_lora_a = lora_forward(w_gate, w_gate_quant_state, w_gate_lora_a, w_gate_lora_b, b_gate, x_norm_2)
        
        # forward process: up_proj
        up, _, up_lora_a = lora_forward(w_up, w_up_quant_state, w_up_lora_a, w_up_lora_b, b_up, x_norm_2)

        #* compress the x_norm_2
        x_norm_2_q, x_norm_2_scale = compression_pack_quant_base(
            x=x_norm_2, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['x_norm_2']
        )
        del x_norm_2
        
        # apply activation function (for gate)
        fn = torch.nn.functional.silu(gate)
        
        #* compress the gate
        gate_q, gate_scale = compression_pack_quant_base(
            x=gate, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['gate']
        )
        del gate
            
        # hadamard
        hadamard = up * fn
        
        #* compress the up / fn
        up_q, up_scale = compression_pack_quant_base(
            x=up, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['up']
        )
        del up
        fn_q, fn_scale = compression_pack_quant_base(
            x=fn, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['fn']
        )
        del fn
            
        # forward process: down_proj
        down, _, down_lora_a = lora_forward(w_down, w_down_quant_state, w_down_lora_a, w_down_lora_b, b_down, hadamard)
        
        #* compress the hadamard
        hadamard_q, hadamard_scale = compression_pack_quant_base(
            x=hadamard, q_bit=q_bit,
            q_method='per-channel', it_num=iteration,
            it_num_thd=iteration_threshold, static_value=static_value['hadamard']
        )
        del hadamard
        
        # residual connection
        x_out = x_medium + down
        del x_medium, down
        
        ctx.save_for_backward(
            ### buffered activation (attention) ###
            x_o, x_q, x_scale, # x
            mean_1, rstd_1, # buffer for rmsnorm
            x_norm_1_q, x_norm_1_scale, # x_norm_1
            cos, sin, # buffer for rope
            q_q, q_scale, q_zero_point, # q
            k_q, k_scale, k_zero_point, # k
            v_q, v_scale, # v
            o_q, o_scale, # o
            q_lora_a, k_lora_a, v_lora_a, # buffer for lora (qkv)
            o_final_lora_a, # buffer for lora (o)
            ### buffered activation (mlp) ###
            mean_2, rstd_2, # buffer for rmsnorm
            x_medium_o, x_medium_q, x_medium_scale, # x_medium
            x_norm_2_q, x_norm_2_scale, # x_norm_2
            gate_q, gate_scale, # gate
            up_q, up_scale, # up
            fn_q, fn_scale, # fn
            hadamard_q, hadamard_scale, # hadamard
            gate_lora_a, up_lora_a, down_lora_a,
            ### weights (attention) ###
            norm_weight_1, norm_bias_1,
            w_q, b_q, w_q_lora_a, w_q_lora_b,
            w_k, b_k, w_k_lora_a, w_k_lora_b,
            w_v, b_v, w_v_lora_a, w_v_lora_b,
            w_o, b_o, w_o_lora_a, w_o_lora_b,
            ### weights (mlp) ###
            norm_weight_2, norm_bias_2,
            w_gate, b_gate, w_gate_lora_a, w_gate_lora_b,
            w_up, b_up, w_up_lora_a, w_up_lora_b,
            w_down, b_down, w_down_lora_a, w_down_lora_b,
        )
        ctx.quant_state = (
            w_q_quant_state,
            w_k_quant_state,
            w_v_quant_state,
            w_o_quant_state,
            w_gate_quant_state,
            w_up_quant_state,
            w_down_quant_state,
        )
        ctx.input_layernorm_channel = x_channel_idx 
        ctx.post_layernorm_channel = x_medium_channel_idx
        ctx.num_heads = num_heads
        ctx.block_size = block_size
        ctx.num_warps = num_warps
        ctx.head_dim = head_dim
        ctx.q_bit = q_bit
        
        ctx.softmax_lse = softmax_lse
        ctx.cu_q_lens = cu_q_lens
        ctx.max_s = max_s
        ctx.rng_state = rng_state
        ctx.softmax_scale = softmax_scale
        ctx.indices = indices # buffer for flash_attn

        return x_out, x_channel_idx, x_scale, \
            x_norm_1_scale, \
            q_scale, q_zero_point, k_scale, k_zero_point, v_scale, o_scale, \
            x_medium_channel_idx, x_medium_scale, \
            x_norm_2_scale, \
            gate_scale, up_scale, fn_scale, hadamard_scale
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, *args):
        (
            w_q_quant_state,
            w_k_quant_state,
            w_v_quant_state,
            w_o_quant_state,
            w_gate_quant_state,
            w_up_quant_state,
            w_down_quant_state,
        ) = ctx.quant_state
        
        (
            ### buffered activation (attention) ###
            x_o, x_q, x_scale, # x
            mean_1, rstd_1, # buffer for rmsnorm
            x_norm_1_q, x_norm_1_scale, # x_norm_1
            cos, sin, # buffer for rope
            q_q, q_scale, q_zero_point, # q
            k_q, k_scale, k_zero_point, # k
            v_q, v_scale, # v
            o_q, o_scale, # o
            q_lora_a, k_lora_a, v_lora_a, # buffer for lora (qkv)
            o_final_lora_a, # buffer for lora (o)
            ### buffered activation (mlp) ###
            mean_2, rstd_2, # buffer for rmsnorm
            x_medium_o, x_medium_q, x_medium_scale, # x_medium
            x_norm_2_q, x_norm_2_scale, # x_norm_2
            gate_q, gate_scale, # gate
            up_q, up_scale, # up
            fn_q, fn_scale, # fn
            hadamard_q, hadamard_scale, # hadamard
            gate_lora_a, up_lora_a, down_lora_a,
            ### weights (attention) ###
            norm_weight_1, norm_bias_1,
            w_q, b_q, w_q_lora_a, w_q_lora_b,
            w_k, b_k, w_k_lora_a, w_k_lora_b,
            w_v, b_v, w_v_lora_a, w_v_lora_b,
            w_o, b_o, w_o_lora_a, w_o_lora_b,
            ### weights (mlp) ###
            norm_weight_2, norm_bias_2,
            w_gate, b_gate, w_gate_lora_a, w_gate_lora_b,
            w_up, b_up, w_up_lora_a, w_up_lora_b,
            w_down, b_down, w_down_lora_a, w_down_lora_b,
        ) = ctx.saved_tensors
        
        #* dequantize hadamard
        hadamard = decompression_dequantization(hadamard_q, hadamard_scale, ctx.q_bit)
        del hadamard_q, hadamard_scale
        
        # down proj part
        grad_w_down_lora_a, grad_w_down_lora_b, grad_down = lora_backward(w_down, w_down_quant_state, w_down_lora_a, w_down_lora_b, hadamard, down_lora_a, grad_output)
        del hadamard
        
        #* dequantize up
        up = decompression_dequantization(up_q, up_scale, ctx.q_bit)
        fn = decompression_dequantization(fn_q, fn_scale, ctx.q_bit)
        del up_q, up_scale, fn_q, fn_scale
        
        # hadamard
        grad_hadamard_1 = grad_down * up
        grad_hadamard_2 = grad_down * fn
        del grad_down, up, fn
        
        #* dequantize gate
        gate = decompression_dequantization(gate_q, gate_scale, ctx.q_bit)
        grad_fn = silu_backward(gate, grad_hadamard_1)
        del gate_q, gate_scale, gate, grad_hadamard_1
        
        #* dequantize x_norm_2
        x_norm_2 = decompression_dequantization(x_norm_2_q, x_norm_2_scale, ctx.q_bit)
        del x_norm_2_q, x_norm_2_scale
        
        # gate proj part
        grad_w_gate_lora_a, grad_w_gate_lora_b, grad_gate = lora_backward(w_gate, w_gate_quant_state, w_gate_lora_a, w_gate_lora_b, x_norm_2, gate_lora_a, grad_fn)
        del grad_fn
        
        # up proj part
        grad_w_up_lora_a, grad_w_up_lora_b, grad_up = lora_backward(w_up, w_up_quant_state, w_up_lora_a, w_up_lora_b, x_norm_2, up_lora_a, grad_hadamard_2)
        grad_gate_up = grad_up + grad_gate
        del grad_up, grad_gate, grad_hadamard_2
        
        #* dequantize x_medium
        x_medium = outlier_addition_fuse_decompression_dequantization(x_medium_q, x_medium_scale, x_medium_o, ctx.post_layernorm_channel, ctx.q_bit)
        del x_medium_q, x_medium_scale, x_medium_o
        
        # layernorm & rmsnorm backward
        grad_norm_2, grad_rmsnorm_2_w, = rmsnorm_backward(
            grad_gate_up, x_medium, norm_weight_2, mean_2, rstd_2, # TODO: other params
            1e-5, ctx.num_warps, ctx.block_size
        )
        del x_medium, grad_gate_up
        
        # residual connection
        grad_medium = grad_norm_2 + grad_output
        
        #* dequantize o
        o = decompression_dequantization(o_q, o_scale, ctx.q_bit)
        del o_q, o_scale
        
        # o part
        grad_w_o_lora_a, grad_w_o_lora_b, grad_o = lora_backward(w_o, w_o_quant_state, w_o_lora_a, w_o_lora_b, o, o_final_lora_a, grad_medium)
        
        # reshape
        grad_o = hidden_to_head_shape(grad_o, ctx.num_heads)
        grad_o = grad_o.transpose(1, 2)[0]
        
        #* dequantize a
        q = decompression_dequantization_with_zero_point(q_q, q_scale, q_zero_point, ctx.q_bit, is_head=True, num_heads=ctx.num_heads)
        k = decompression_dequantization_with_zero_point(k_q, k_scale, k_zero_point, ctx.q_bit, is_head=True, num_heads=ctx.num_heads)
        v = decompression_dequantization(v_q, v_scale, ctx.q_bit, is_head=True, num_heads=ctx.num_heads)

        #! flash_attn begin
        out_padded = hidden_to_head_shape(o, ctx.num_heads)
        out_padded = rearrange(out_padded, "b h s d -> b s h d")[0]
        q = rearrange(q, "b h s d -> b s h d")[0]
        k = rearrange(k, "b h s d -> b s h d")[0]
        v = rearrange(v, "b h s d -> b s h d")[0]
        o = hidden_to_head_shape(o, ctx.num_heads)
        o = rearrange(o, "b s h d -> b h s d")[0]
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        _flash_attn_varlen_backward(
            dout=grad_o, q=q, k=k, v=v,
            out=o, softmax_lse=ctx.softmax_lse,
            dq=dqkv[:, 0], dk=dqkv[:, 1], dv=dqkv[:, 2],
            cu_seqlens_q=ctx.cu_q_lens, cu_seqlens_k=ctx.cu_q_lens,
            max_seqlen_q=ctx.max_s, max_seqlen_k=ctx.max_s,
            dropout_p=0.0, softmax_scale=ctx.softmax_scale,
            causal=True, window_size=(-1, -1), 
            alibi_slopes=None, deterministic=False, rng_state=ctx.rng_state
        )
        dqkv = dqkv[..., : grad_o.shape[-1]]
        grad_q, grad_k, grad_v = dqkv[:, 0], dqkv[:, 1], dqkv[:, 2]
        
        grad_q, grad_k, grad_v = grad_q.transpose(0, 1), grad_k.transpose(0, 1), grad_v.transpose(0, 1)
        # expand batch dimension
        grad_q, grad_k, grad_v = grad_q.unsqueeze(0).contiguous(), grad_k.unsqueeze(0).contiguous(), grad_v.unsqueeze(0).contiguous()
        #! flash_attn end

        BLOCK_SIZE, num_warps = calculate_settings(ctx.head_dim // 2)
        N_GROUPS = 128
        grad_q = rope_backward(grad_q.transpose(1, 2), cos, sin, N_GROUPS, BLOCK_SIZE, num_warps).transpose(1, 2) # TODO: other params
        grad_k = rope_backward(grad_k.transpose(1, 2), cos, sin, N_GROUPS, BLOCK_SIZE, num_warps).transpose(1, 2)

        grad_q = head_to_hidden_shape(grad_q)
        grad_k = head_to_hidden_shape(grad_k)
        grad_v = head_to_hidden_shape(grad_v)
        
        #* dequantize x_norm_1
        x_norm_1 = decompression_dequantization(x_norm_1_q, x_norm_1_scale, ctx.q_bit)
        del x_norm_1_q, x_norm_1_scale
        
        # backward of q_proj
        grad_w_q_lora_a, grad_w_q_lora_b, grad_x = lora_backward(w_q, w_q_quant_state, w_q_lora_a, w_q_lora_b, x_norm_1, q_lora_a, grad_q)

        # backward of k_proj
        grad_w_k_lora_a, grad_w_k_lora_b, grad_x_temp = lora_backward(w_k, w_k_quant_state, w_k_lora_a, w_k_lora_b, x_norm_1, k_lora_a, grad_k)
        grad_x += grad_x_temp

        # backward of v_proj
        grad_w_v_lora_a, grad_w_v_lora_b, grad_x_temp = lora_backward(w_v, w_v_quant_state, w_v_lora_a, w_v_lora_b, x_norm_1, v_lora_a, grad_v)
        grad_x += grad_x_temp
        
        #* dequantize x
        x = outlier_addition_fuse_decompression_dequantization(x_q, x_scale, x_o, ctx.input_layernorm_channel, ctx.q_bit)
        del x_q, x_scale, x_o
        
        # layernorm or rmsnorm backward
        grad_norm_1, grad_rmsnorm_1_w, = rmsnorm_backward(
            grad_x, x, norm_weight_1, mean_1, rstd_1, # TODO: other params
            1e-5, ctx.num_warps, ctx.block_size
        )
            
        # residual connection
        grad_input = grad_norm_1 + grad_medium
        
        return (
            grad_input,
            #############attention part#############
            grad_rmsnorm_1_w,
            None,
            ####################################
            None,
            None,
            ####################################
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
            #############mlp part#############
            grad_rmsnorm_2_w,
            None,
            ####################################
            None,
            None,
            None,
            grad_w_gate_lora_a,
            grad_w_gate_lora_b,
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
            grad_w_down_lora_b
        ) + (None,) * 9


class FusedLlamaLayerIntra(torch.nn.Module):
    def __init__(
        self,
    ):
        super(FusedLlamaLayerIntra, self).__init__()
        self.iteration = 0
        self.static_value = {
            'x': {'outlier_channel_index': None, 'scale': None},
            'x_norm_1': {'scale': None},
            'q': {'scale': None, 'zero_point': None},
            'k': {'scale': None, 'zero_point': None},
            'v': {'scale': None},
            'o': {'scale': None},
            'x_medium': {'outlier_channel_index': None, 'scale': None},
            'x_norm_2': {'scale': None},
            'gate': {'scale': None},
            'up': {'scale': None},
            'fn': {'scale': None},
            'hadamard': {'scale': None},
        }
    
        
    def set_hyclora_config(self, hyclora_config):
        self.hyclora_config = hyclora_config
        self.use_hyclora = hyclora_config.use_hyclora
        self.iteration_threshold = hyclora_config.iteration_threshold
        self.layernorm_outlier_ratio = hyclora_config.layernorm_outlier_ratio
        self.q_bit = hyclora_config.q_bit
    
        
    def forward(
        self,
        input: torch.Tensor,
        ############################################
        norm_weight_1: torch.Tensor,
        norm_bias_1: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
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
        ############################################
        norm_weight_2: torch.Tensor,
        norm_bias_2: torch.Tensor,
        gate_proj_base: bnb.nn.modules.Linear4bit,
        gate_proj_lora_a: torch.nn.Linear,
        gate_proj_lora_b: torch.nn.Linear,
        up_proj_base: bnb.nn.modules.Linear4bit,
        up_proj_lora_a: torch.nn.Linear,
        up_proj_lora_b: torch.nn.Linear,
        down_proj_base: bnb.nn.modules.Linear4bit,
        down_proj_lora_a: torch.nn.Linear,
        down_proj_lora_b: torch.nn.Linear,
        ############################################
        attention_mask: torch.Tensor,
        num_heads: int,
        head_dim: int,
    ):
        y, x_channel_idx, x_scale, \
        x_norm_1_scale, \
        q_scale, q_zero_point, k_scale, k_zero_point, v_scale, o_scale, \
        x_medium_channel_idx, x_medium_scale, \
        x_norm_2_scale, \
        gate_scale, up_scale, fn_scale, hadamard_scale = FusedLlamaLayerIntraFunc.apply(
            input,
            #############attention part#############
            norm_weight_1,
            norm_bias_1,
            ####################################
            cos,
            sin,
            ####################################
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
            #############mlp part#############
            norm_weight_2,
            norm_bias_2,
            ####################################
            gate_proj_base.weight,
            gate_proj_base.bias,
            gate_proj_base.weight.quant_state,
            gate_proj_lora_a.default.weight.T,
            gate_proj_lora_b.default.weight.T,
            ####################################
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
            attention_mask,
            num_heads,
            head_dim,
            ####################################
            self.iteration,
            self.iteration_threshold,
            self.static_value,
            self.layernorm_outlier_ratio,
            self.q_bit,
        )
        
        if self.iteration < self.iteration_threshold:
            self.static_value['x'] = update_dict(self.static_value['x'], {'outlier_channel_index': x_channel_idx, 'scale': x_scale}, self.iteration)
            self.static_value['x_norm_1'] = update_dict(self.static_value['x_norm_1'], {'scale': x_norm_1_scale}, self.iteration)
            self.static_value['q'] = update_dict(self.static_value['q'], {'scale': q_scale, 'zero_point': q_zero_point}, self.iteration)
            self.static_value['k'] = update_dict(self.static_value['k'], {'scale': k_scale, 'zero_point': k_zero_point}, self.iteration)
            self.static_value['v'] = update_dict(self.static_value['v'], {'scale': v_scale}, self.iteration)
            self.static_value['o'] = update_dict(self.static_value['o'], {'scale': o_scale}, self.iteration)
            self.static_value['x_medium'] = update_dict(self.static_value['x_medium'], {'outlier_channel_index': x_medium_channel_idx, 'scale': x_medium_scale}, self.iteration)
            self.static_value['x_norm_2'] = update_dict(self.static_value['x_norm_2'], {'scale': x_norm_2_scale}, self.iteration)
            self.static_value['gate'] = update_dict(self.static_value['gate'], {'scale': gate_scale}, self.iteration)
            self.static_value['up'] = update_dict(self.static_value['up'], {'scale': up_scale}, self.iteration)
            self.static_value['fn'] = update_dict(self.static_value['fn'], {'scale': fn_scale}, self.iteration)
            self.static_value['hadamard'] = update_dict(self.static_value['hadamard'], {'scale': hadamard_scale}, self.iteration)
            
        self.iteration += 1
        
        return y
        