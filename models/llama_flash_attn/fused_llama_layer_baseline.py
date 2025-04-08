import math
import torch
import typing
import bitsandbytes as bnb
import torch.nn.functional as F
import bitsandbytes.functional as BF

from operators.rope_kernels import rope_forward, rope_backward, calculate_settings
from operators.silu_kernels import silu_backward
from operators.rmsnorm_updatable_kernels import rmsnorm_backward, rmsnorm_forward

from ..utils.compute_utils import(
    hidden_to_head_shape, head_to_hidden_shape,
    lora_forward, lora_backward,
    unpad_input, pad_input
)

from einops import rearrange
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward, _flash_attn_varlen_backward


class FusedLlamaLayerBaselineFunc(torch.autograd.Function):
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
    ):
        bsz, q_len, _ = x.size()
        # layernorm or rmsnorm
        x_norm_1, mean_1, rstd_1, _, _ = rmsnorm_forward(x, norm_weight_1, eps = 1e-5)

        # compute q,k,v
        # forward process: q_proj
        q, _, q_lora_a = lora_forward(w_q, w_q_quant_state, w_q_lora_a, w_q_lora_b, b_q, x_norm_1)

        # forward process: k_proj
        k, _, k_lora_a = lora_forward(w_k, w_k_quant_state, w_k_lora_a, w_k_lora_b, b_k, x_norm_1)

        # forward process: v_proj
        v, _, v_lora_a = lora_forward(w_v, w_v_quant_state, w_v_lora_a, w_v_lora_b, b_v, x_norm_1)
        
        # reshape
        q = hidden_to_head_shape(q, num_heads)
        k = hidden_to_head_shape(k, num_heads)
        v = hidden_to_head_shape(v, num_heads)
        
        ctx.q_shape = q.shape

        # TODO: apply positional encoding
        q = rope_forward(q.transpose(1, 2), cos, sin).transpose(1, 2)
        k = rope_forward(k.transpose(1, 2), cos, sin).transpose(1, 2)
        
        #! flash_attn begin
        qkv = torch.stack([q, k, v], dim=2)
        qkv = qkv.transpose(1, 3)
        
        key_padding_mask = attention_mask
        nheads = qkv.shape[-2]
        qkv = rearrange(qkv, "b s three h d -> b s (three h d)")

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
        
        # residual connection
        x_medium = x + o_final
        
        # layernorm or rmsnorm
        x_norm_2, mean_2, rstd_2, block_size, num_warps = rmsnorm_forward(x_medium, norm_weight_2, eps = 1e-5)
        
        # forward process: gate_proj
        gate, _, gate_lora_a = lora_forward(w_gate, w_gate_quant_state, w_gate_lora_a, w_gate_lora_b, b_gate, x_norm_2)
        
        # forward process: up_proj
        up, _, up_lora_a = lora_forward(w_up, w_up_quant_state, w_up_lora_a, w_up_lora_b, b_up, x_norm_2)
        
        # apply activation function (for gate)
        fn = torch.nn.functional.silu(gate)
            
        # hadamard
        hadamard = up * fn
            
        # forward process: down_proj
        down, _, down_lora_a = lora_forward(w_down, w_down_quant_state, w_down_lora_a, w_down_lora_b, b_down, hadamard)
        
        # residual connection
        x_out = x_medium + down
        
        ctx.save_for_backward(
            ### activations (attention) ###
            x, # buffer for rmsnorm
            mean_1, # buffer for rmsnorm
            rstd_1, # buffer for rmsnorm
            x_norm_1, # buffer for lora (qkv)
            q_lora_a, # buffer for lora (qkv)
            k_lora_a, # buffer for lora (qkv)
            v_lora_a, # buffer for lora (qkv)
            cos, # buffer for rope
            sin, # buffer for rope
            q, # buffer for flash_attn
            k, # buffer for flash_attn
            v, # buffer for flash_attn
            out_padded, # buffer for flash_attn
            o_final_lora_a, # buffer for lora (o)
            ### activations (mlp) ###
            x_medium, # buffer for rmsnorm
            mean_2, # buffer for rmsnorm
            rstd_2, # buffer for rmsnorm
            x_norm_2, # buffer for lora (gate/up)
            gate_lora_a, # buffer for lora (gate)
            gate, # buffer for gelu/silu
            up_lora_a, # buffer for lora (up)
            up, # buffer for hadamard
            fn, # buffer for hadamard
            hadamard, # buffer for down
            down_lora_a, # buffer for lora (down)
            ### weights (attention) ###
            norm_weight_1, 
            norm_bias_1,
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
            ### weights (mlp) ###
            norm_weight_2,
            norm_bias_2,
            #**********************
            w_gate,
            b_gate,
            w_gate_lora_a,
            w_gate_lora_b,
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
        ctx.num_heads = num_heads
        ctx.block_size = block_size
        ctx.num_warps = num_warps
        ctx.head_dim = head_dim
        
        ctx.softmax_lse = softmax_lse
        ctx.cu_q_lens = cu_q_lens
        ctx.max_s = max_s
        ctx.rng_state = rng_state
        ctx.softmax_scale = softmax_scale
        ctx.indices = indices # buffer for flash_attn

        return x_out
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
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
            ### activations (attention) ###
            x, # buffer for rmsnorm
            mean_1, # buffer for rmsnorm
            rstd_1, # buffer for rmsnorm
            x_norm_1, # buffer for lora (qkv)
            q_lora_a, # buffer for lora (qkv)
            k_lora_a, # buffer for lora (qkv)
            v_lora_a, # buffer for lora (qkv)
            cos, # buffer for rope
            sin, # buffer for rope
            q, # buffer for flash_attn
            k, # buffer for flash_attn
            v, # buffer for flash_attn
            out_padded, # buffer for flash_attn
            o_final_lora_a, # buffer for lora (o)
            ### activations (mlp) ###
            x_medium, # buffer for rmsnorm
            mean_2, # buffer for rmsnorm
            rstd_2, # buffer for rmsnorm
            x_norm_2, # buffer for lora (gate/up)
            gate_lora_a, # buffer for lora (gate)
            gate, # buffer for gelu/silu
            up_lora_a, # buffer for lora (up)
            up, # buffer for hadamard
            fn, # buffer for hadamard
            hadamard, # buffer for down
            down_lora_a, # buffer for lora (down)
            ### weights (attention) ###
            norm_weight_1, 
            norm_bias_1,
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
            ### weights (mlp) ###
            norm_weight_2,
            norm_bias_2,
            #**********************
            w_gate,
            b_gate,
            w_gate_lora_a,
            w_gate_lora_b,
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
        ) = ctx.saved_tensors
        
        # down proj part
        grad_w_down_lora_a, grad_w_down_lora_b, grad_down = lora_backward(w_down, w_down_quant_state, w_down_lora_a, w_down_lora_b, hadamard, down_lora_a, grad_output)
        
        # hadamard
        grad_hadamard_1 = grad_down * up
        grad_hadamard_2 = grad_down * fn
        
        # TODO: activation backward
        grad_fn = silu_backward(gate, grad_hadamard_1)
        
        # gate proj part
        grad_w_gate_lora_a, grad_w_gate_lora_b, grad_gate = lora_backward(w_gate, w_gate_quant_state, w_gate_lora_a, w_gate_lora_b, x_norm_2, gate_lora_a, grad_fn)
        
        # up proj part
        grad_w_up_lora_a, grad_w_up_lora_b, grad_up = lora_backward(w_up, w_up_quant_state, w_up_lora_a, w_up_lora_b, x_norm_2, up_lora_a, grad_hadamard_2)
        grad_gate_up = grad_up + grad_gate
        
        # layernorm & rmsnorm backward
        grad_norm_2, grad_rmsnorm_2_w = rmsnorm_backward(
            grad_gate_up, x_medium, norm_weight_2, mean_2, rstd_2, # TODO: other params
            1e-5, ctx.num_warps, ctx.block_size
        )
        
        # residual connection
        grad_medium = grad_norm_2 + grad_output
        
        # o part
        bsz, q_len, _ = x.size()
        o = rearrange(pad_input(rearrange(out_padded, "nnz h d -> nnz (h d)"), ctx.indices, bsz, q_len), "b s (h d) -> b s h d", h=ctx.num_heads)
        o = o.view(bsz, q_len, ctx.num_heads * ctx.head_dim)
        grad_w_o_lora_a, grad_w_o_lora_b, grad_o = lora_backward(w_o, w_o_quant_state, w_o_lora_a, w_o_lora_b, o, o_final_lora_a, grad_medium)
        
        # reshape
        grad_o = hidden_to_head_shape(grad_o, ctx.num_heads)
        grad_o = grad_o.transpose(1, 2)[0]
        
        #! flash_attn begin
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        _flash_attn_varlen_backward(
            dout=grad_o, q=q, k=k, v=v,
            out=out_padded, softmax_lse=ctx.softmax_lse,
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
        
        grad_q = rope_backward(grad_q.transpose(1, 2).contiguous(), cos, sin, N_GROUPS, BLOCK_SIZE, num_warps).transpose(1, 2).contiguous() # TODO: other params
        grad_k = rope_backward(grad_k.transpose(1, 2).contiguous(), cos, sin, N_GROUPS, BLOCK_SIZE, num_warps).transpose(1, 2).contiguous()

        grad_q = head_to_hidden_shape(grad_q)
        grad_k = head_to_hidden_shape(grad_k)
        grad_v = head_to_hidden_shape(grad_v)
        
        # backward of q_proj
        grad_w_q_lora_a, grad_w_q_lora_b, grad_x = lora_backward(w_q, w_q_quant_state, w_q_lora_a, w_q_lora_b, x_norm_1, q_lora_a, grad_q)

        # backward of k_proj
        grad_w_k_lora_a, grad_w_k_lora_b, grad_x_temp = lora_backward(w_k, w_k_quant_state, w_k_lora_a, w_k_lora_b, x_norm_1, k_lora_a, grad_k)
        grad_x += grad_x_temp

        # backward of v_proj
        grad_w_v_lora_a, grad_w_v_lora_b, grad_x_temp = lora_backward(w_v, w_v_quant_state, w_v_lora_a, w_v_lora_b, x_norm_1, v_lora_a, grad_v)
        grad_x += grad_x_temp
        
        # layernorm or rmsnorm backward
        grad_norm_1, grad_rmsnorm_1_w = rmsnorm_backward(
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
        ) + (None,) * 3


class FusedLlamaLayerBaseline(torch.nn.Module):
    def __init__(
        self,
    ):
        super(FusedLlamaLayerBaseline, self).__init__()
        self.iteration = 0
        self.static_value = None
        
    def set_hyclora_config(self, hyclora_config):
        self.use_hyclora = hyclora_config.use_hyclora
        self.hyclora_config = hyclora_config
        
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
        y = FusedLlamaLayerBaselineFunc.apply(
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
        )
        
        return y
        