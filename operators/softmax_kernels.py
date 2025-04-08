import torch
import torch.nn.functional as F
import triton
import triton.language as tl

@triton.jit
def softmax_backward_kernel(
    y_ptr, g_ptr, o_ptr,
    b_stride, h_stride, row_stride, col_stride,
    b, h, n_rows, n_cols, 
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_b = tl.program_id(axis=1)
    offs_h = tl.program_id(axis=2)
    num_pid_m = tl.cdiv(n_rows, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(n_cols, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    y_row_ptr = y_ptr + row_stride * offs_m[:, None] + col_stride * offs_n[None, :] + b_stride * offs_b + h_stride * offs_h
    g_row_ptr = g_ptr + row_stride * offs_m[:, None] + col_stride * offs_n[None, :] + b_stride * offs_b + h_stride * offs_h
    mask = (offs_b < b) & (offs_h < h) & (offs_m[:, None] < n_rows) & (offs_n[None, :] < n_cols) 
    
    y = tl.load(y_row_ptr, mask=mask)
    g = tl.load(g_row_ptr, mask=mask)
    y_g = y * g
    y_g_sum = tl.sum(y_g)
    o = (g - y_g_sum) * y
    
    o_row_ptr = o_ptr + row_stride * offs_m[:, None] + col_stride * offs_n[None, :] + b_stride * offs_b + h_stride * offs_h
    tl.store(o_row_ptr, o, mask=mask)
    
    
def softmax_backward(y, grad_y):    
    b, h, n, _ = y.shape
    block_size_modified = triton.next_power_of_2(n)
    o = torch.empty_like(y)
    
    assert grad_y.shape == y.shape
    grid = lambda META: (
        triton.cdiv(n, META['BLOCK_SIZE_M']) * triton.cdiv(n, META['BLOCK_SIZE_N']), b, h
    )
    # Create a number of persistent programs.
    softmax_backward_kernel[grid](
        y, grad_y, o, 
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        b, h, n, n, 
        BLOCK_SIZE_M=1, BLOCK_SIZE_N=block_size_modified, GROUP_SIZE_M=1
    )
    return o