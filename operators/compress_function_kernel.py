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
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=1, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=2, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=4, num_warps=4),
        
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=1, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=2, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=4, num_warps=8),
        
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=4, num_warps=4),
        
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=4, num_warps=8),
        
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=1, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=2, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=4, num_warps=4),
        
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=1, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=2, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 4,}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def compression_quantization_kernel(
    # Pointers to matrices
    x_ptr, q_ptr, s_ptr,
    # Matrix dimensions
    B, M, N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_lm` is how much to increase `l_ptr`
    # by to get the element one row down (A has M rows).
    stride_xb, stride_xm, stride_xn,
    stride_qb, stride_qm, stride_qn,
    stride_sn, # scaling factor has no stride in the m dimension
    # compress params
    quantize_bit: tl.constexpr, elem_per_position: tl.constexpr, 
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr, 
):
    """
    o_ptr means outlier, q_ptr means quantized values, s_ptr means scaling factors
    """
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

    offs_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    x_ptrs = x_ptr + stride_xm * offs_xm[:, None] + stride_xn * offs_xn[None, :] + offs_b * stride_xb
    x_mask = (offs_b < B) & (offs_xm[:, None] < M) & (offs_xn[None, :] < N)
     
    offs_qm = offs_xm
    offs_qn = pid_n * (BLOCK_SIZE_N // elem_per_position) + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    q_ptrs = q_ptr + stride_qm * offs_qm[:, None] + stride_qn * offs_qn[None, :] + offs_b * stride_qb
    q_mask = (offs_b < B) & (offs_qm[:, None] < M) & (offs_qn[None, :] < N // elem_per_position)
    q = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N // elem_per_position), dtype=tl.uint8)
    
    offs_sn = offs_xn
    s_ptrs = s_ptr + stride_sn * offs_sn[None, :] 
    s_mask = (offs_sn[None, :] < N)
    
    max_val = 2 ** (quantize_bit - 1) - 1
    min_val = -2 ** (quantize_bit - 1)
    total_range = 2 ** (quantize_bit - 1)
    
    for i in range(elem_per_position):
        x_ptrs_new = x_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        s_ptrs_new = s_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
      
        x = tl.load(x_ptrs_new, mask=x_mask, other=0.0)
        s = tl.load(s_ptrs_new, mask=s_mask, other=1.0)
        
        x = tl_math.round(x / s)
        x = tl.where(x < min_val, min_val, x)
        x = tl.where(x > max_val, max_val, x)
        x = x + total_range

        element_int = tl_math.float2uint_rn(x.to(tl.float32)).to(tl.uint8)
        # element_int = bfloat162uint_rn(x).to(tl.uint8)
        q |= (element_int << (quantize_bit * i)).to(tl.uint8)
    
    tl.store(q_ptrs, q, mask=q_mask)

    
@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=1, num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=2, num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=3, num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=4, num_warps=2),
        
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=1, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=2, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=4, num_warps=4),
        
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=1, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=2, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=4, num_warps=8),
        
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=4, num_warps=2),
        
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=4, num_warps=4),
        
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def decompression_dequantization_kernel(
    # Pointers to matrices
    x_ptr, q_ptr, s_ptr,
    # Matrix dimensions
    B, M, N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_lm` is how much to increase `l_ptr`
    # by to get the element one row down (A has M rows).
    stride_xb, stride_xm, stride_xn,
    stride_qb, stride_qm, stride_qn,
    stride_sn, # scaling factor has no stride in the m dimension
    # compress params
    quantize_bit: tl.constexpr, elem_per_position: tl.constexpr, 
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr, 
):
    # basic pointers
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

    offs_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    x_ptrs = x_ptr + stride_xm * offs_xm[:, None] + stride_xn * offs_xn[None, :] + offs_b * stride_xb
    x_mask = (offs_b < B) & (offs_xm[:, None] < M) & (offs_xn[None, :] < N)
    
    offs_sn = offs_xn
    s_ptrs = s_ptr + stride_sn * offs_sn[None, :]
    s_mask = (offs_sn[None, :] < N)

    offs_qm = offs_xm
    offs_qn = pid_n * (BLOCK_SIZE_N // elem_per_position) + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    q_ptrs = q_ptr + stride_qm * offs_qm[:, None] + stride_qn * offs_qn[None, :] + offs_b * stride_qb
    q_mask = (offs_b < B) & (offs_qm[:, None] < M) & (offs_qn[None, :] < N // elem_per_position)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.uint8)

    mask = (1 << quantize_bit) - 1

    # extract the quantized values
    for i in range(elem_per_position):
        x_ptrs_new = x_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        s_ptrs_new = s_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        s = tl.load(s_ptrs_new, mask=s_mask, other=1.0)
        x = tl_math.uint2float_rn((q & mask).to(tl.uint32))
        x = x.to(tl.bfloat16)
        x = x - 2 ** (quantize_bit - 1)
        x = x * s
        x = x.to(tl.bfloat16)
        tl.store(x_ptrs_new, x, mask=x_mask)
        q = (q >> quantize_bit).to(tl.uint8)
        
        
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=4, num_warps=2),
        
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=4, num_warps=4),
        
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def compression_quantization_with_zero_point_kernel(
    # Pointers to matrices
    x_ptr, q_ptr, s_ptr, z_ptr,
    # Matrix dimensions
    B, M, N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_lm` is how much to increase `l_ptr`
    # by to get the element one row down (A has M rows).
    stride_xb, stride_xm, stride_xn,
    stride_qb, stride_qm, stride_qn,
    stride_sn, # scaling factor has no stride in the m dimension
    stride_zn,
    # compress params
    quantize_bit: tl.constexpr, elem_per_position: tl.constexpr, 
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr, 
):
    """
    o_ptr means outlier, q_ptr means quantized values, s_ptr means scaling factors
    """
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

    offs_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    x_ptrs = x_ptr + stride_xm * offs_xm[:, None] + stride_xn * offs_xn[None, :] + offs_b * stride_xb
    x_mask = (offs_b < B) & (offs_xm[:, None] < M) & (offs_xn[None, :] < N)
     
    offs_qm = offs_xm
    offs_qn = pid_n * (BLOCK_SIZE_N // elem_per_position) + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    q_ptrs = q_ptr + stride_qm * offs_qm[:, None] + stride_qn * offs_qn[None, :] + offs_b * stride_qb
    q_mask = (offs_b < B) & (offs_qm[:, None] < M) & (offs_qn[None, :] < N // elem_per_position)
    q = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N // elem_per_position), dtype=tl.uint8)
    
    offs_sn = offs_xn
    s_ptrs = s_ptr + stride_sn * offs_sn[None, :] 
    s_mask = (offs_sn[None, :] < N)
    
    offs_zn = offs_xn
    z_ptrs = z_ptr + stride_zn * offs_zn[None, :]
    z_mask = (offs_zn[None, :] < N)
    
    max_val = 2 ** (quantize_bit - 1) - 1
    min_val = -2 ** (quantize_bit - 1)
    total_range = 2 ** (quantize_bit - 1)
    
    for i in range(elem_per_position):
        x_ptrs_new = x_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        s_ptrs_new = s_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        z_ptrs_new = z_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
      
        x = tl.load(x_ptrs_new, mask=x_mask, other=0.0)
        s = tl.load(s_ptrs_new, mask=s_mask, other=1.0)
        z = tl.load(z_ptrs_new, mask=z_mask, other=0.0)
        
        x = tl_math.round(x / s + z)
        x = tl.where(x < min_val, min_val, x)
        x = tl.where(x > max_val, max_val, x)
        x = x + total_range

        element_int = tl_math.float2uint_rn(x.to(tl.float32)).to(tl.uint8)
        # element_int = bfloat162uint_rn(x).to(tl.uint8)
        q |= (element_int << (quantize_bit * i)).to(tl.uint8)
    
    tl.store(q_ptrs, q, mask=q_mask)
    
    
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=4, num_warps=2),
        
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=4, num_warps=4),
        
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8,}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def decompression_dequantization_with_zero_point_kernel(
    # Pointers to matrices
    x_ptr, q_ptr, s_ptr, z_ptr,
    # Matrix dimensions
    B, M, N,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_lm` is how much to increase `l_ptr`
    # by to get the element one row down (A has M rows).
    stride_xb, stride_xm, stride_xn,
    stride_qb, stride_qm, stride_qn,
    stride_sn, # scaling factor has no stride in the m dimension
    stride_zn,
    # compress params
    quantize_bit: tl.constexpr, elem_per_position: tl.constexpr, 
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr, 
):
    # basic pointers
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

    offs_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    x_ptrs = x_ptr + stride_xm * offs_xm[:, None] + stride_xn * offs_xn[None, :] + offs_b * stride_xb
    x_mask = (offs_b < B) & (offs_xm[:, None] < M) & (offs_xn[None, :] < N)
    
    offs_sn = offs_xn
    s_ptrs = s_ptr + stride_sn * offs_sn[None, :]
    s_mask = (offs_sn[None, :] < N)
    
    offs_zn = offs_xn
    z_ptrs = z_ptr + stride_zn * offs_zn[None, :]
    z_mask = (offs_zn[None, :] < N)

    offs_qm = offs_xm
    offs_qn = pid_n * (BLOCK_SIZE_N // elem_per_position) + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    q_ptrs = q_ptr + stride_qm * offs_qm[:, None] + stride_qn * offs_qn[None, :] + offs_b * stride_qb
    q_mask = (offs_b < B) & (offs_qm[:, None] < M) & (offs_qn[None, :] < N // elem_per_position)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.uint8)

    mask = (1 << quantize_bit) - 1

    # extract the quantized values
    for i in range(elem_per_position):
        x_ptrs_new = x_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        s_ptrs_new = s_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        z_ptrs_new = z_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        
        s = tl.load(s_ptrs_new, mask=s_mask, other=1.0)
        z = tl.load(z_ptrs_new, mask=z_mask, other=0.0)
        x = tl_math.uint2float_rn((q & mask).to(tl.uint32))
        x = x.to(tl.bfloat16)
        x = x - 2 ** (quantize_bit - 1)
        x = (x - z) * s
        x = x.to(tl.bfloat16)
        tl.store(x_ptrs_new, x, mask=x_mask)
        q = (q >> quantize_bit).to(tl.uint8)