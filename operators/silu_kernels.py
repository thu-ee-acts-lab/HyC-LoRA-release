import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4, }, num_stages=4, num_warps=8)
    ],
    key=['M', 'N'],
)
@triton.jit
def silu_backward_kernel(
    x_ptr, g_ptr, o_ptr,
    B, M, N,
    stride_xb, stride_xm, stride_xn,
    stride_gb, stride_gm, stride_gn,
    stride_ob, stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_b = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x_ptrs = x_ptr + stride_xm * offs_m[:, None] + stride_xn * offs_n[None, :] + offs_b * stride_xb
    x_mask = (offs_b < B) & (offs_m[:, None] < M) & (offs_n[None, :] < N)
    g_ptrs = g_ptr + stride_gm * offs_m[:, None] + stride_gn * offs_n[None, :] + offs_b * stride_gb
    g_mask = x_mask
    o_ptrs = o_ptr + stride_om * offs_m[:, None] + stride_on * offs_n[None, :] + offs_b * stride_ob
    o_mask = x_mask
    
    x = tl.load(x_ptrs, mask=x_mask)
    g = tl.load(g_ptrs, mask=g_mask)
    x_sigmoid = tl.sigmoid(x.to(tl.float32)).to(tl.bfloat16)
    o = g * (x_sigmoid * (1 + x * (1 - x_sigmoid)))
    
    tl.store(o_ptrs, o, mask=o_mask)
    
    
def silu_backward(x, g):
    B, M, N = x.shape
    assert g.shape == x.shape
    
    o = torch.empty_like(x)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    silu_backward_kernel[grid](
        x, g, o,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        g.stride(0), g.stride(1), g.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
    )
    return o


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8, }, num_stages=2, num_warps=2), #! config: 1bit, 2; 2bit, 2; 4bit, 4; 8bit, 4
    ],
    key=['M', 'N'],
)
@triton.jit
def triton_silu_hadamard_kernel(
    x1_ptr, x2_ptr, o_ptr,
    B, M, N,
    stride_xb, stride_xm, stride_xn,
    stride_gb, stride_gm, stride_gn,
    stride_ob, stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_b = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x1_ptrs = x1_ptr + stride_xm * offs_m[:, None] + stride_xn * offs_n[None, :] + offs_b * stride_xb
    x_mask = (offs_b < B) & (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x2_ptrs = x2_ptr + stride_gm * offs_m[:, None] + stride_gn * offs_n[None, :] + offs_b * stride_gb
    g_mask = x_mask
    o_ptrs = o_ptr + stride_om * offs_m[:, None] + stride_on * offs_n[None, :] + offs_b * stride_ob
    o_mask = x_mask
    
    x1 = tl.load(x1_ptrs, mask=x_mask)
    x2 = tl.load(x2_ptrs, mask=g_mask)
    x_silu = tl.sigmoid(x1.to(tl.float32)).to(tl.bfloat16) * x1
    o = x_silu * x2
    
    tl.store(o_ptrs, o, mask=o_mask)
    

# apply silu to x1, then mult with x2
def silu_hadamard(x, g):
    B, M, N = x.shape
    assert g.shape == x.shape
    
    o = torch.empty_like(x)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    triton_silu_hadamard_kernel[grid](
        x, g, o,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        g.stride(0), g.stride(1), g.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
    )
    return o