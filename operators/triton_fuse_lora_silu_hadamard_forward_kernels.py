import torch
import triton
import triton.language as tl
from packaging.version import Version

triton_version = triton.__version__
if Version(triton_version) < Version("3.0.0"):
    import triton.language.math as tl_math
else:
    import triton.language.extra.cuda.libdevice as tl_math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=4, num_warps=4),
        
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def triton_fuse_silu_forward_kernel(
    xa1_ptr, b1_ptr, x1_ptr, q1_ptr, s1_ptr,
    xa2_ptr, b2_ptr, x2_ptr, q2_ptr, s2_ptr, 
    o_ptr,
    B, M, N,
    stride_xab, stride_xam, stride_xar,
    stride_br, stride_bn,
    stride_xb, stride_xm, stride_xn,
    stride_qb, stride_qm, stride_qn,
    stride_sn,
    # compress params
    quantize_bit: tl.constexpr, elem_per_position: tl.constexpr, 
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, 
):
    pid = tl.program_id(axis=0)
    offs_b = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_xam = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    
    offs_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    
    x1_ptrs = x1_ptr + stride_xm * offs_xm[:, None] + stride_xn * offs_xn[None, :] + offs_b * stride_xb
    x1_mask = (offs_b < B) & (offs_xm[:, None] < M) & (offs_xn[None, :] < N)
    x2_ptrs = x2_ptr + stride_xm * offs_xm[:, None] + stride_xn * offs_xn[None, :] + offs_b * stride_xb
    x2_mask = (offs_b < B) & (offs_xm[:, None] < M) & (offs_xn[None, :] < N)
    
    offs_qm = offs_xm
    offs_qn = pid_n * (BLOCK_SIZE_N // elem_per_position) + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    q1_ptrs = q1_ptr + stride_qm * offs_qm[:, None] + stride_qn * offs_qn[None, :] + offs_b * stride_qb
    q2_ptrs = q2_ptr + stride_qm * offs_qm[:, None] + stride_qn * offs_qn[None, :] + offs_b * stride_qb
    q1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N // elem_per_position), dtype=tl.uint8)
    q2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N // elem_per_position), dtype=tl.uint8)
    
    offs_sn = offs_xn
    s1_ptrs = s1_ptr + stride_sn * offs_sn[None, :] 
    s1_mask = (offs_sn[None, :] < N)
    s2_ptrs = s2_ptr + stride_sn * offs_sn[None, :]
    s2_mask = (offs_sn[None, :] < N)
    
    offs_ym = offs_xm
    offs_yn = offs_xn
    
    offs_r = tl.arange(0, BLOCK_SIZE_K)
    xa1_ptrs = xa1_ptr + (offs_xam[:, None] * stride_xam + offs_r[None, :] * stride_xar) + offs_b * stride_xab
    xa1 = tl.load(xa1_ptrs)
    xa2_ptrs = xa2_ptr + (offs_xam[:, None] * stride_xam + offs_r[None, :] * stride_xar) + offs_b * stride_xab
    xa2 = tl.load(xa2_ptrs)
    
    b1_ptrs = b1_ptr + (offs_r[:, None] * stride_br + offs_bn[None, :] * stride_bn)
    b2_ptrs = b2_ptr + (offs_r[:, None] * stride_br + offs_bn[None, :] * stride_bn)
    
    o_ptrs = o_ptr + stride_xm * offs_ym[:, None] + stride_xn * offs_yn[None, :] + offs_b * stride_xb
    
    max_val = 2 ** (quantize_bit - 1) - 1
    min_val = -2 ** (quantize_bit - 1)
    total_range = 2 ** (quantize_bit - 1)
    
    for i in range(elem_per_position):
        ################### x1 ###################
        x1_ptrs_new = x1_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        s1_ptrs_new = s1_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        b1_ptrs_new = b1_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        b2_ptrs_new = b2_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        o_ptrs_new = o_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        
        # step 1: quantize x
        x1 = tl.load(x1_ptrs_new, mask=x1_mask, other=0.0)
        s1 = tl.load(s1_ptrs_new, mask=s1_mask, other=1.0)
        b1 = tl.load(b1_ptrs_new)
        b2 = tl.load(b2_ptrs_new)
        
        x1_quant = tl_math.round(x1 / s1)
        x1_quant = tl.where(x1_quant < min_val, min_val, x1_quant)
        x1_quant = tl.where(x1_quant > max_val, max_val, x1_quant)
        x1_quant = x1_quant + total_range
        
        element_int1 = tl_math.float2uint_rn(x1_quant.to(tl.float32)).to(tl.uint8)
        q1 |= (element_int1 << (quantize_bit * i)).to(tl.uint8)
        
        # step 2: add by lora
        y1 = x1 + tl.dot(xa1, b1).to(tl.bfloat16)
        
        # step 3: silu
        y1 = (y1 * tl.sigmoid(y1.to(tl.float32)).to(tl.bfloat16))
        
        ################### x2 ###################
        x2_ptrs_new = x2_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        s2_ptrs_new = s2_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        
        # step 1: quantize x
        x2 = tl.load(x2_ptrs_new, mask=x2_mask, other=0.0)
        s2 = tl.load(s2_ptrs_new, mask=s2_mask, other=1.0)
        
        x2_quant = tl_math.round(x2 / s2)
        x2_quant = tl.where(x2_quant < min_val, min_val, x2_quant)
        x2_quant = tl.where(x2_quant > max_val, max_val, x2_quant)
        x2_quant = x2_quant + total_range
        
        element_int2 = tl_math.float2uint_rn(x2_quant.to(tl.float32)).to(tl.uint8)
        q2 |= (element_int2 << (quantize_bit * i)).to(tl.uint8)
        
        # step 2: add by lora
        y2 = x2 + tl.dot(xa2, b2).to(tl.bfloat16)
        
        # step 3: hadamard
        o = y1 * y2
        
        tl.store(o_ptrs_new, o)
        
    tl.store(q1_ptrs, q1)
    tl.store(q2_ptrs, q2)
    
    
def triton_fuse_silu_forward(xa1, b1, x1, s1, xa2, b2, x2, s2, quantize_bit=8):
    B, M, N = x1.shape
    R = b1.shape[0]
    
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    elem_per_position = 8 // quantize_bit
    q1 = torch.empty((B, M, N // elem_per_position), device=x1.device, dtype=torch.uint8)
    q2 = torch.empty((B, M, N // elem_per_position), device=x1.device, dtype=torch.uint8)
    o = torch.empty((B, M, N), device=x1.device, dtype=torch.bfloat16)
    
    #! must keep b1, b2 contiguous (the rest part of the code is not optimized for non-contiguous tensors)
    b1 = b1.contiguous()
    b2 = b2.contiguous()
    
    triton_fuse_silu_forward_kernel[grid](
        xa1, b1, x1, q1, s1,
        xa2, b2, x2, q2, s2,
        o,
        B, M, N,
        xa1.stride(0), xa1.stride(1), xa1.stride(2),
        b1.stride(0), b1.stride(1),
        x1.stride(0), x1.stride(1), x1.stride(2),
        q1.stride(0), q1.stride(1), q1.stride(2),
        s1.stride(1),
        quantize_bit, elem_per_position, BLOCK_SIZE_K=R,
    )
    
    return q1, q2, o