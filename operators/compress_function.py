import torch
from .compress_function_kernel import *

# ********************utils********************

def hidden_to_head_shape(x: torch.Tensor, num_heads: int):
    bsz, seq_len, hidden_dim = x.shape
    head_dim = hidden_dim // num_heads
    return x.reshape(bsz, seq_len, num_heads, head_dim).transpose(1, 2)

def head_to_hidden_shape(x: torch.Tensor):
    bsz, _, seq_len, _ = x.shape
    return x.transpose(1, 2).reshape(bsz, seq_len, -1)

def update_dict(old_dict, new_dict, iteration):
    for name in old_dict.keys():
        if name == 'outlier_channel_index':
            if old_dict[name] is None:
                old_dict[name] = new_dict[name]
            else: # TODO merge logic
                old_dict[name] = new_dict[name]
        else:
            if old_dict[name] is None:
                old_dict[name] = new_dict[name]
            else:
                old_dict[name] = (iteration * old_dict[name] + new_dict[name]) / (iteration + 1)
    return old_dict


def get_statistics_softmax(x: torch.Tensor, outlier_ratio: float):
    outlier = torch.kthvalue(x.float().flatten(), int(x.numel() * (1 - outlier_ratio))).values
    return outlier

def get_statistics_outlier(x: torch.Tensor, outlier_ratio: float):
    outlier = torch.kthvalue(x.float().abs().flatten(), int(x.numel() * (1 - outlier_ratio))).values
    return outlier


def get_statistics_structed_pruning(x: torch.Tensor, outlier_ratio: float):
    channel_norm = x.abs().norm(dim=-2)
    outlier_channel_index = torch.topk(channel_norm, int(x.shape[-1] * outlier_ratio), largest=True).indices
    return outlier_channel_index


def get_statistics_only_quant(x: torch.Tensor, q_bit: int = 8, q_method: str = 'per-tensor'):
    if len(x.shape) == 4:
        batch, num_head, seq_len, sep_dim = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch, seq_len, num_head * sep_dim)

    x_sample = x[0]
    if q_method == 'per-tensor':
        scale = (x_sample.max() - x_sample.min()) / (2 ** q_bit - 1)
    elif q_method == 'per-channel':
        scale = (x_sample.max(dim=-2, keepdim=True).values - x_sample.min(dim=-2, keepdim=True).values) / (2 ** q_bit - 1)
    else:
        raise "Unsupport Quantize Method"
    
    del x
    return scale.to(torch.bfloat16)


def get_statistics_only_quant_zero_point(x: torch.Tensor, q_bit: int = 8, q_method: str = 'per-tensor'):
    if len(x.shape) == 4:
        batch, num_head, seq_len, sep_dim = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch, seq_len, num_head * sep_dim)

    x_sample = x[0]
    if q_method == 'per-tensor':
        scale = (x_sample.max() - x_sample.min()) / (2 ** q_bit - 1)
    elif q_method == 'per-channel':
        scale = (x_sample.max(dim=-2, keepdim=True).values - x_sample.min(dim=-2, keepdim=True).values) / (2 ** q_bit - 1)
        zero_point = -torch.round(x_sample.min(dim=-2, keepdim=True).values / scale) - (2 ** (q_bit - 1))
    else:
        raise "Unsupport Quantize Method"
    
    del x
    return scale.to(torch.bfloat16), zero_point.to(torch.int8)


def get_statistics_channel_base(x: torch.Tensor, outlier_ratio: float, q_bit: int = 8, q_method: str = 'per-tensor'):
    x_ = x.clone()
    if len(x_.shape) == 4:
        batch, num_head, seq_len, sep_dim = x_.shape
        x_ = x_.permute(0, 2, 1, 3).reshape(batch, seq_len, num_head * sep_dim)
    
    channel_norm = x_[0].abs().norm(dim=-2)
    outlier_channel_index = torch.topk(channel_norm, int(x_[0].shape[-1] * outlier_ratio), largest=True).indices

    x_outlier = x_[:, :, outlier_channel_index]
    x_outlier = x_outlier.to(torch.bfloat16)
    x_[:, :, outlier_channel_index] = 0

    x_sub_outlier = x_[0]
    if q_method == 'per-tensor':
        scale = (x_sub_outlier.max() - x_sub_outlier.min()) / (2 ** q_bit - 1)
    elif q_method == 'per-channel':
        scale = (x_sub_outlier.max(dim=-2, keepdim=True).values - x_sub_outlier.min(dim=-2, keepdim=True).values) / (2 ** q_bit - 1)
    else:
        raise "Unsupport Quantize Method"
    
    del x_
    
    # the scale factor should not be zero(but since the corresponding channel is removed, there's no impact on final result)
    scale += (scale == 0) * 1e-3
    return outlier_channel_index, scale.to(torch.bfloat16)


# ********************compress/decompress interface********************

# outlier subtraction + quantization
def outlier_subtraction_fuse_compression_quantization(x, s, channel, quantize_bit=8, dtype=torch.bfloat16):
    # Change shape if need
    is_head = len(x.shape) == 4
    if is_head:
        x = head_to_hidden_shape(x)

    # decide dtype
    x, s = x.to(dtype), s.to(dtype)
    B, M, N = x.shape

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    elem_per_position = 8 // quantize_bit
    q = torch.empty((B, M, N // elem_per_position), device=x.device, dtype=torch.uint8)
    
    # remove outlier channels
    x_outlier = x[:, :, channel]
    x[:, :, channel] = 0
    
    # quantize the rest part
    compression_quantization_kernel[grid](
        x, q, s,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(1),
        quantize_bit, elem_per_position
    )

    del x
    return x_outlier, q


# outlier addition + dequantization
def outlier_addition_fuse_decompression_dequantization(q, s, x_outlier, channel, quantize_bit=8, is_head=False, num_heads=1, dtype=torch.bfloat16):
    B, M, _ = q.shape
    N = s.shape[-1]

    # dtype
    s = s.to(dtype)

    # 1D launch kernel where each block gets its own program.
    elem_per_position = 8 // quantize_bit
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    x = torch.empty((B, M, N), device=q.device, dtype=torch.bfloat16)

    decompression_dequantization_kernel[grid](
        x, q, s,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(1),
        quantize_bit, elem_per_position,
    )
    x[:, :, channel] = x_outlier
    
    if is_head:
        x = hidden_to_head_shape(x, num_heads=num_heads)
    
    return x


# quantization (only with scale)
def compression_quantization(x, s, quantize_bit=8, dtype=torch.bfloat16):
    # Change shape if need
    is_head = len(x.shape) == 4
    if is_head:
        x = head_to_hidden_shape(x)

    # decide dtype
    x, s = x.to(dtype), s.to(dtype)
    B, M, N = x.shape

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    elem_per_position = 8 // quantize_bit
    q = torch.empty((B, M, N // elem_per_position), device=x.device, dtype=torch.uint8)
    
    # ************************** triton version **************************
    # quantize the rest part
    compression_quantization_kernel[grid](
        x, q, s,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(1),
        quantize_bit, elem_per_position
    )
    
    # ************************** torch version **************************
    # q = torch.clamp(torch.round(x / s), -2 ** (quantize_bit - 1), 2 ** (quantize_bit - 1) - 1).to(torch.int8)
    
    del x
    return q


# dequantization (only with scale)
def decompression_dequantization(q, s, quantize_bit=8, is_head=False, num_heads=1, dtype=torch.bfloat16):
    B, M, _ = q.shape
    N = s.shape[-1]

    # dtype
    s = s.to(dtype)

    # 1D launch kernel where each block gets its own program.
    elem_per_position = 8 // quantize_bit
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    x = torch.empty((B, M, N), device=q.device, dtype=torch.bfloat16)

    # ************************** triton version **************************
    decompression_dequantization_kernel[grid](
        x, q, s,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(1),
        quantize_bit, elem_per_position,
    )
    
    if is_head:
        x = hidden_to_head_shape(x, num_heads=num_heads)
    
    return x


# quantization (with scale and zero point)
def compression_quantization_with_zero_point(x, s, z, quantize_bit=8, dtype=torch.bfloat16):
    # Change shape if need
    is_head = len(x.shape) == 4
    if is_head:
        x = head_to_hidden_shape(x)

    # decide dtype
    x, s, z = x.to(dtype), s.to(dtype), z.to(dtype)
    B, M, N = x.shape

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    elem_per_position = 8 // quantize_bit
    q = torch.empty((B, M, N // elem_per_position), device=x.device, dtype=torch.uint8)
    
    # quantize the rest part
    compression_quantization_with_zero_point_kernel[grid](
        x, q, s, z, 
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(1),
        z.stride(1),
        quantize_bit, elem_per_position
    )

    del x
    return q


# dequantization (with scale and zero point)
def decompression_dequantization_with_zero_point(q, s, z, quantize_bit=8, is_head=False, num_heads=1, dtype=torch.bfloat16):
    B, M, _ = q.shape
    N = s.shape[-1]

    # dtype
    s = s.to(dtype)
    z = z.to(dtype)

    # 1D launch kernel where each block gets its own program.
    elem_per_position = 8 // quantize_bit
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    x = torch.empty((B, M, N), device=q.device, dtype=torch.bfloat16)

    decompression_dequantization_with_zero_point_kernel[grid](
        x, q, s, z,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(1),
        z.stride(1),
        quantize_bit, elem_per_position,
    )
    
    if is_head:
        x = hidden_to_head_shape(x, num_heads=num_heads)
    
    return x

# compress softmax's attention map
def compression_softmax(x: torch.Tensor, outlier: float):
    mask = (x > outlier)
    x_outlier = x * mask
    x_outlier_sparse = x_outlier.flatten().to_sparse()
    return x_outlier_sparse

# decompress softmax's attention map
def decompression_softmax(x_sparse: torch.Tensor):
    return x_sparse.to_dense()

# unstructed compression(use COO format)
def compression_unstructed_pruning_coo(x: torch.Tensor, outlier: float):
    mask = (x.abs() > outlier)
    x_outlier = x * mask
    x_outlier_sparse = x_outlier.to_sparse()
    return x_outlier_sparse

# unstructed decompress(use COO format)
def decompression_unstructed_pruning_coo(x_sparse: torch.Tensor):
    return x_sparse.to_dense()

# unstructed compression(use bitmap format)
def compression_unstructed_pruning_bitmap(x: torch.Tensor, outlier: float):
    mask = (x.abs() > outlier)
    x_outlier = x * mask
    x_outlier_sparse = x_outlier[mask]
    return x_outlier_sparse, mask

# unstructed decompress(use bitmap format)
def decompression_unstructed_pruning_bitmap(x_sparse: torch.Tensor, mask: torch.Tensor):
    x = torch.empty_like(mask, device=x_sparse.device, dtype=x_sparse.dtype)
    x[mask] = x_sparse
    return x


# ********************pack "get statistics" and "compress"********************
# type: outlier channel extraction + quantization
def compression_pack_channel_base(x, o_ratio, q_bit, q_method, it_num, it_num_thd, static_value):
    if it_num < it_num_thd:
        o_channel_idx, scale = get_statistics_channel_base(x, o_ratio, q_bit, q_method)
    else:
        o_channel_idx, scale = static_value['outlier_channel_index'], static_value['scale']
    o, q = outlier_subtraction_fuse_compression_quantization(x, scale, o_channel_idx, q_bit)
    return o, q, o_channel_idx, scale

# type: quantization
def compression_pack_quant_base(x, q_bit, q_method, it_num, it_num_thd, static_value):
    if it_num < it_num_thd:
        scale = get_statistics_only_quant(x, q_bit, q_method)
    else:
        scale = static_value['scale']
    q = compression_quantization(x, scale, q_bit)
    return q, scale

# type: quantization with zero point
def compression_pack_quant_zp_base(x, q_bit, q_method, it_num, it_num_thd, static_value):
    if it_num < it_num_thd:
        scale, zero_point = get_statistics_only_quant_zero_point(x, q_bit, q_method)
    else:
        scale, zero_point = static_value['scale'], static_value['zero_point']
    q = compression_quantization_with_zero_point(x, scale, zero_point, q_bit)
    return q, scale, zero_point

# type: softmax attention map compression
def compression_pack_softmax_base(x, o_ratio, it_num, it_num_thd, static_value):
    if it_num < it_num_thd:
        outlier = get_statistics_softmax(x, o_ratio)
    else:
        outlier = static_value['outlier']
    o = compression_softmax(x, outlier)
    return o, outlier

# type: unstructed pruning(using COO format)
def compression_pack_unstructed_pruning_coo(x, o_ratio, it_num, it_num_thd, static_value):
    if it_num < it_num_thd:
        outlier = get_statistics_outlier(x, o_ratio) # a funny reuse
    else:
        outlier = static_value['outlier']
    x_outlier = compression_unstructed_pruning_coo(x, outlier)
    return x_outlier, outlier

# type: unstructed pruning(using bitmap format)
def compression_pack_unstructed_pruning_bitmap(x, o_ratio, it_num, it_num_thd, static_value):
    if it_num < it_num_thd:
        outlier = get_statistics_outlier(x, o_ratio) # a funny reuse
    else:
        outlier = static_value['outlier']
    x_outlier, mask = compression_unstructed_pruning_bitmap(x, outlier)
    return x_outlier, outlier, mask
    