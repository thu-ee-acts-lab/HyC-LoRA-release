# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
import triton.language as tl

MAX_FUSED_SIZE = 65536
ROPE_GROUP_SIZE = 4

def calculate_settings(n):
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps = 4
    if BLOCK_SIZE >= 32768: 
        num_warps = 32
    elif BLOCK_SIZE >= 8192: 
        num_warps = 16
    elif BLOCK_SIZE >= 2048: 
        num_warps = 8
    return BLOCK_SIZE, num_warps


@triton.heuristics({"BACKWARD_PASS": lambda args: args["BACKWARD_PASS"],})
@triton.jit
def _rope_embedding(
    Q,     Q_row_stride,
    cos, cos_row_stride,
    sin, sin_row_stride,
    seqlen,
    head_dim      : tl.constexpr,
    n_heads       : tl.constexpr,
    BACKWARD_PASS : tl.constexpr,
    BLOCK_SIZE    : tl.constexpr,
):
    ROPE_GROUP_SIZE = 4
    row_position  = tl.program_id(0)
    group_head_position = tl.program_id(1)
    col_offsets  = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    sin1 = tl.load(sin + (row_position % seqlen)*sin_row_stride + \
                   half_head_dim*0 + col_offsets, mask = mask, other = 0)
    cos1 = tl.load(cos + (row_position % seqlen)*cos_row_stride + \
                   half_head_dim*0 + col_offsets, mask = mask, other = 0)

    if BACKWARD_PASS:
        # See our blog post for more info.
        sin1 = -sin1

    # [TODO] Autotune ROPE_GROUP_SIZE to be 1, 2, 4, 8
    head_start = group_head_position * ROPE_GROUP_SIZE
    head_end = min((head_start + ROPE_GROUP_SIZE), n_heads)

    # 10% Faster kernel from [HuyNguyen-hust](https://github.com/unslothai/unsloth/pull/238)
    for k in range(head_start, head_end):
        offs_q1 = row_position * Q_row_stride + k * head_dim + col_offsets
        offs_q2 = row_position * Q_row_stride + k * head_dim + col_offsets + half_head_dim

        # For Gemma - sometimes RoPE must be done in float32 and not bfloat16
        Q1 = tl.load(Q + offs_q1, mask = mask, other = 0).to(sin1.dtype)
        Q2 = tl.load(Q + offs_q2, mask = mask, other = 0).to(sin1.dtype)

        tl.store(Q + offs_q1, Q1*cos1 - Q2*sin1, mask = mask)
        tl.store(Q + offs_q2, Q2*cos1 + Q1*sin1, mask = mask)
        
        
def rope_forward(x, cos, sin):
    x = x.contiguous()
    cos, sin = cos.squeeze(), sin.squeeze()
    batch, seq_len, n_heads, head_dim = x.shape
    x = x.view(batch * seq_len, n_heads * head_dim)
    n_rows, n_cols = x.shape
    assert(seq_len <= cos.shape[0])

    # [TODO] Changing blocksize to head_dim//2 seems to have
    # some concurrency / un-deterministic issues.
    BLOCK_SIZE, num_warps = calculate_settings(head_dim // 2) # (head_dim//2)
    
    # group_size = 4 # 4 or 8, too large group_size can hurt performance.
    div, mod = divmod(n_heads, ROPE_GROUP_SIZE)
    n_groups = div + (mod != 0)

    _rope_embedding[(n_rows, n_groups)](
        x, x.stride(0),
        cos, cos.stride(0),
        sin, sin.stride(0),
        seq_len,
        head_dim, n_heads,
        BACKWARD_PASS = False,
        BLOCK_SIZE = BLOCK_SIZE,
        num_warps = num_warps,
    )
    return x.view(batch, seq_len, n_heads, head_dim)


def rope_backward(dy, cos, sin, n_groups, BLOCK_SIZE, num_warps):
    batch, seq_len, n_heads, head_dim = dy.shape
    cos, sin = cos.squeeze(), sin.squeeze()
    dy = dy.reshape(batch * seq_len, n_heads * head_dim)
    # Must be reshape not view
    n_rows, n_cols = dy.shape

    _rope_embedding[(n_rows, n_groups)](
        dy,  dy.stride(0),
        cos, cos.stride(0),
        sin, sin.stride(0),
        seq_len, head_dim, n_heads,
        BACKWARD_PASS = True,
        BLOCK_SIZE = BLOCK_SIZE,
        num_warps = num_warps,
    )
    dy = dy.view(batch, seq_len, n_heads, head_dim)
    return dy