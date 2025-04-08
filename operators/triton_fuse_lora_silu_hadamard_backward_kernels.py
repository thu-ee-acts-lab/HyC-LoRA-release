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
def triton_fuse_silu_backward_kernel(
    xa1_ptr, b1_ptr, q1_ptr, s1_ptr,
    xa2_ptr, b2_ptr, q2_ptr, s2_ptr,
    grad_o_ptr, o_ptr,
    grad_gate_ptr, grad_up_ptr,
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
    
    offs_qm = offs_xm
    offs_qn = pid_n * (BLOCK_SIZE_N // elem_per_position) + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    q1_ptrs = q1_ptr + stride_qm * offs_qm[:, None] + stride_qn * offs_qn[None, :] + offs_b * stride_qb
    q2_ptrs = q2_ptr + stride_qm * offs_qm[:, None] + stride_qn * offs_qn[None, :] + offs_b * stride_qb
    q1 = tl.load(q1_ptrs).to(tl.uint8)
    q2 = tl.load(q2_ptrs).to(tl.uint8)
    
    offs_sn = offs_xn
    s1_ptrs = s1_ptr + stride_sn * offs_sn[None, :] 
    s2_ptrs = s2_ptr + stride_sn * offs_sn[None, :]
    
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
    grad_o_ptrs = grad_o_ptr + stride_xm * offs_ym[:, None] + stride_xn * offs_yn[None, :] + offs_b * stride_xb
    
    grad_gate_ptrs = grad_gate_ptr + stride_xm * offs_ym[:, None] + stride_xn * offs_yn[None, :] + offs_b * stride_xb
    grad_up_ptrs = grad_up_ptr + stride_xm * offs_ym[:, None] + stride_xn * offs_yn[None, :] + offs_b * stride_xb
    
    mask = (1 << quantize_bit) - 1
    total_range = 2 ** (quantize_bit - 1)
    
    for i in range(elem_per_position):
        s1_ptrs_new = s1_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        s2_ptrs_new = s2_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        b1_ptrs_new = b1_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        b2_ptrs_new = b2_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        o_ptrs_new = o_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        grad_o_ptrs_new = grad_o_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        grad_gate_ptrs_new = grad_gate_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        grad_up_ptrs_new = grad_up_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        
        b1 = tl.load(b1_ptrs_new)
        b2 = tl.load(b2_ptrs_new)
        
        # step 1: dequantize x
        s1 = tl.load(s1_ptrs_new)
        x1 = tl_math.uint2float_rn((q1 & mask).to(tl.uint32))
        x1 = x1.to(tl.bfloat16)
        x1 = x1 - total_range
        x1 = x1 * s1
        x1 = x1.to(tl.bfloat16)
        q1 = (q1 >> quantize_bit).to(tl.uint8)
        
        s2 = tl.load(s2_ptrs_new)
        x2 = tl_math.uint2float_rn((q2 & mask).to(tl.uint32))
        x2 = x2.to(tl.bfloat16)
        x2 = x2 - total_range
        x2 = x2 * s2
        x2 = x2.to(tl.bfloat16)
        q2 = (q2 >> quantize_bit).to(tl.uint8)
        
        # step 2: add by lora
        y1 = x1 + tl.dot(xa1, b1).to(tl.bfloat16)
        y2 = x2 + tl.dot(xa2, b2).to(tl.bfloat16)
        
        # step 3: silu
        y1_sigmoid = tl.sigmoid(y1.to(tl.float32)).to(tl.bfloat16)
        y1_silu = (y1 * y1_sigmoid).to(tl.bfloat16)
        
        # step 4: hadamard
        o = y1_silu * y2
        
        tl.store(o_ptrs_new, o)
        
        # step 5: backward (hadamard)
        grad_o = tl.load(grad_o_ptrs_new)
        grad_hadamard_1 = grad_o * y2
        grad_hadamard_2 = grad_o * y1_silu
        
        # step 6: backward (silu)
        grad_silu_1 = grad_hadamard_1 * (y1_sigmoid * (1. + y1 * (1. - y1_sigmoid).to(tl.bfloat16)).to(tl.bfloat16))
        
        # step 7: save grad
        grad_gate = grad_silu_1
        grad_up = grad_hadamard_2
        
        tl.store(grad_gate_ptrs_new, grad_gate)
        tl.store(grad_up_ptrs_new, grad_up)
        
        
def triton_fuse_silu_backward(xa1, b1, q1, s1, xa2, b2, q2, s2, grad_o, quantize_bit=8):
    B, M, N = grad_o.shape
    R = b1.shape[0]
    
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    elem_per_position = 8 // quantize_bit
    
    o = torch.empty((B, M, N), device=grad_o.device, dtype=torch.bfloat16)
    grad_silu = torch.empty((B, M, N), device=grad_o.device, dtype=torch.bfloat16)
    grad_hadamard_2 = torch.empty((B, M, N), device=grad_o.device, dtype=torch.bfloat16)
        
    triton_fuse_silu_backward_kernel[grid](
        xa1, b1, q1, s1,
        xa2, b2, q2, s2,
        grad_o, o,
        grad_silu, grad_hadamard_2,
        B, M, N,
        xa1.stride(0), xa1.stride(1), xa1.stride(2),
        b1.stride(0), b1.stride(1),
        grad_o.stride(0), grad_o.stride(1), grad_o.stride(2),
        q1.stride(0), q1.stride(1), q1.stride(2),
        s1.stride(1),
        quantize_bit, elem_per_position, BLOCK_SIZE_K=R,
    )
    
    return o, grad_silu, grad_hadamard_2