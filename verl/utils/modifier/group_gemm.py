# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from .utils import TIMING_LOGGER, is_hip

# TODO: int64 offsets for large tensors (>=2**31) on MI300


@triton.jit
def _silu(x):
    return x * (1 / (1 + tl.exp(-x)))


@triton.jit
def _dsilu(x):
    x_sigmoid = 1 / (1 + tl.exp(-x))
    return x_sigmoid * (1 + x * (1 - x_sigmoid))


@triton.jit
def _act_fake_quant(x, N, BITS: tl.constexpr, IS_HIP: tl.constexpr):
    if BITS == 8:
        s = 127 / tl.max(tl.abs(x), axis=-1, keep_dims=True)
    elif BITS == 4:
        s = 2.6457513110645907 * N / tl.sum(tl.abs(x), axis=-1, keep_dims=True)
    elif BITS == 2:
        s = 1.7320508075688772 * N / tl.sum(tl.abs(x), axis=-1, keep_dims=True)
    if IS_HIP:
        x = tl.extra.hip.libdevice.llrint(x * s)
    else:
        x = tl.extra.cuda.libdevice.rint(x * s)
    if BITS == 8:
        x = tl.clamp(x, -128, 127) / s
    elif BITS == 4:
        x = tl.clamp(x, -8, 7) / s
    elif BITS == 2:
        x = tl.clamp(x, -4, 3) / s
    return x


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "waves_per_eu": 1,
                "matrix_instr_nonkdim": 16,
                "kpack": 1,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "waves_per_eu": 1,
                "matrix_instr_nonkdim": 16,
                "kpack": 1,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 1,
                "waves_per_eu": 1,
                "matrix_instr_nonkdim": 16,
                "kpack": 1,
            },
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 1,
                "waves_per_eu": 1,
                "matrix_instr_nonkdim": 16,
                "kpack": 1,
            },
            num_warps=8,
            num_stages=2,
        ),
    ]
    if is_hip()
    else [
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4
        ),
    ],
    key=["N", "K", "TRANSPOSE_B"],
)
@triton.jit
def _grouped_gemm_sparse_M_kernel(
    a_ptr,  # [M, K] or [T * M, K]
    b_ptr,  # [E, N, K]
    c_ptr,  # [T * M, N]
    cnt_ptr,  # [E] in [0, M)
    idx_ptr,  # [E, M] in [0, T * M)
    M,
    N,
    K,
    S,  # Start expert index
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_ie,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    FORKS_A: tl.constexpr,
    SPARSE_A: tl.constexpr,
    SPARSE_C: tl.constexpr,
    TRANSPOSE_B: tl.constexpr,  # For auto-tune
):
    eid = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.num_programs(axis=0) // num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offset_s = tl.load(cnt_ptr + (S - 1)) if S > 0 else 0
    global_eid = S + eid
    start_m = 0 if eid == 0 else tl.load(cnt_ptr + global_eid - 1) - offset_s
    end_m = tl.load(cnt_ptr + global_eid) - offset_s

    if start_m + pid_m * BLOCK_SIZE_M >= end_m:
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    mask = start_m + offs_m < end_m

    if SPARSE_A:
        idx = tl.load(idx_ptr + global_eid * stride_ie + offs_m, mask=mask, other=0)
    else:
        idx = start_m + offs_m
    a_ptrs = a_ptr + ((idx // FORKS_A)[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + eid * stride_be + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=mask[:, None], other=0.0)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(c_ptr.type.element_ty)

    if not SPARSE_A and SPARSE_C:
        idx = tl.load(idx_ptr + global_eid * stride_ie + offs_m, mask=mask, other=0)
    elif SPARSE_A and not SPARSE_C:
        idx = start_m + offs_m
    c_ptrs = c_ptr + (idx[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, c, mask=mask[:, None])


def grouped_gemm_sparse_M(
    A: torch.Tensor,  # [M, K] or [T * M, K]
    B: torch.Tensor,  # [E, K, N]
    cnt: torch.Tensor,  # [E, ]
    idx: torch.Tensor,  # [E, M]
    M: int,
    N: int,
    K: int,
    E: int,
    T: int,
    S: int,
    max_M_per_E: int,
    forks_A: int,
    sparse_A: bool,
    sparse_C: bool,
    transpose_B: bool,
):
    if sparse_C:  # num_tokens (M) is used to create final output
        C = torch.empty((M * T, N), device=A.device, dtype=A.dtype)
    else:  # num_local_tokens (M) is used to create intermediate output
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    if transpose_B:
        stride_BN, stride_BK = B.stride(1), B.stride(2)
    else:
        stride_BK, stride_BN = B.stride(1), B.stride(2)
    _grouped_gemm_sparse_M_kernel[
        lambda META: (
            triton.cdiv(max_M_per_E, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            E,
        )
    ](
        A,
        B,
        C,
        cnt,
        idx,
        max_M_per_E,
        N,
        K,
        S,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        stride_BK,
        stride_BN,
        C.stride(0),
        C.stride(1),
        idx.stride(0) if idx is not None else 0,
        FORKS_A=forks_A,
        SPARSE_A=sparse_A,
        SPARSE_C=sparse_C,
        TRANSPOSE_B=transpose_B,
    )
    return C


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 16,
                "GROUP_SIZE_M": 1,
                "waves_per_eu": 1,
                "matrix_instr_nonkdim": 32,
            },
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 16,
                "GROUP_SIZE_M": 4,
                "waves_per_eu": 1,
                "matrix_instr_nonkdim": 32,
            },
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 16,
                "GROUP_SIZE_M": 1,
                "waves_per_eu": 2,
                "matrix_instr_nonkdim": 32,
            },
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 16,
                "GROUP_SIZE_M": 4,
                "waves_per_eu": 2,
                "matrix_instr_nonkdim": 32,
            },
            num_warps=8,
            num_stages=4,
        ),
    ]
    if is_hip()
    else [
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=4
        ),
    ],
    key=["M", "N", "SPARSE_A", "SPARSE_B"],
)
@triton.jit
def _grouped_gemm_sparse_K_kernel(
    a_ptr,  # [M, T * K]
    b_ptr,  # [T * K, N]
    c_ptr,  # [E, M, N]
    cnt_ptr,  # [E] in [0, K)
    idx_ptr,  # [E, K] in [0, T * K)
    M,
    N,
    K,
    S,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ce,
    stride_cm,
    stride_cn,
    stride_ie,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPARSE_A: tl.constexpr,
    FORKS_A: tl.constexpr,
    SPARSE_B: tl.constexpr,
    FORKS_B: tl.constexpr,
):
    eid = tl.program_id(axis=1)
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offset_s = tl.load(cnt_ptr + (S - 1)) if S > 0 else 0
    global_eid = S + eid
    start_k = 0 if eid == 0 else tl.load(cnt_ptr + global_eid - 1) - offset_s
    end_k = tl.load(cnt_ptr + global_eid) - offset_s

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am
    b_ptrs = b_ptr + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    k_off = 0
    for k in range(0, (end_k - start_k) // BLOCK_SIZE_K):
        if SPARSE_A:
            sparse_idx = tl.load(idx_ptr + global_eid * stride_ie + k_off + offs_k)
            a_idx = sparse_idx // FORKS_A
            if SPARSE_B:
                b_idx = sparse_idx // FORKS_B
            else:
                b_idx = start_k + k_off + offs_k
        else:
            a_idx = start_k + k_off + offs_k
            if SPARSE_B:
                sparse_idx = tl.load(idx_ptr + global_eid * stride_ie + k_off + offs_k)
                b_idx = sparse_idx // FORKS_B
            else:
                b_idx = start_k + k_off + offs_k
        a = tl.load(a_ptrs + a_idx[None, :] * stride_ak)
        b = tl.load(b_ptrs + b_idx[:, None] * stride_bk)
        accumulator += tl.dot(a, b)
        k_off += BLOCK_SIZE_K
    if start_k + k_off < end_k:
        mask = start_k + k_off + offs_k < end_k
        if SPARSE_A:
            sparse_idx = tl.load(idx_ptr + global_eid * stride_ie + k_off + offs_k, mask=mask, other=0)
            a_idx = sparse_idx // FORKS_A
            if SPARSE_B:
                b_idx = sparse_idx // FORKS_B
            else:
                b_idx = start_k + k_off + offs_k
        else:
            a_idx = start_k + k_off + offs_k
            if SPARSE_B:
                sparse_idx = tl.load(idx_ptr + global_eid * stride_ie + k_off + offs_k, mask=mask, other=0)
                b_idx = sparse_idx // FORKS_B
            else:
                b_idx = start_k + k_off + offs_k
        a = tl.load(a_ptrs + a_idx[None, :] * stride_ak, mask=mask[None, :], other=0.0)
        b = tl.load(b_ptrs + b_idx[:, None] * stride_bk, mask=mask[:, None], other=0.0)
        accumulator += tl.dot(a, b)
    c = accumulator.to(c_ptr.type.element_ty)

    c_ptrs = c_ptr + eid * stride_ce + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c)


def grouped_gemm_sparse_K(
    A: torch.Tensor,  # [T * K, M]
    B: torch.Tensor,  # [T * K, N]
    cnt: torch.Tensor,  # [E, ]
    idx: torch.Tensor,  # [E, M]
    M: int,
    N: int,
    K: int,
    E: int,
    T: int,
    S: int,
    sparse_A: bool,
    sparse_B: bool,
    forks_A: int,
    forks_B: int,
):
    C = torch.empty((E, M, N), device=A.device, dtype=A.dtype)
    _grouped_gemm_sparse_K_kernel[
        lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            E,
        )
    ](
        A,
        B,
        C,
        cnt,
        idx,
        M,
        N,
        K,
        S,
        A.stride(1),
        A.stride(0),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        idx.stride(0) if idx is not None else 0,
        SPARSE_A=sparse_A,
        SPARSE_B=sparse_B,
        FORKS_A=forks_A,
        FORKS_B=forks_B,
    )
    return C


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 1}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_M": 4}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_M": 8}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_M": 16}, num_warps=4, num_stages=1),
    ],
    key=["N"],
)
@triton.jit
def _fused_silu_kernel(
    x_ptr,
    y_ptr,
    cnt_ptr,
    rmsw_ptr,
    eps,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_we,
    stride_wn,
    M,
    N,
    E,
    S,
    NORM_QUANT: tl.constexpr,
    QUANT_BITS: tl.constexpr,
    IS_HIP: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    eid = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=0)

    offset_s = tl.load(cnt_ptr + (S - 1)) if S > 0 else 0
    global_eid = S + eid
    start_m = 0 if global_eid == 0 else tl.load(cnt_ptr + global_eid - 1) - offset_s
    end_m = tl.load(cnt_ptr + global_eid) - offset_s
    if start_m + pid_m * BLOCK_SIZE_M >= end_m:
        return

    offs_m = start_m + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    mask_m = offs_m < end_m
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    x1_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x3_ptrs = x1_ptrs + N * stride_xn
    x1 = tl.load(x1_ptrs, mask=mask, other=0.0).to(tl.float32)
    x3 = tl.load(x3_ptrs, mask=mask, other=0.0).to(tl.float32)

    y = _silu(x1) * x3
    if NORM_QUANT:
        # RMS Norm
        w_ptrs = rmsw_ptr + eid * stride_we + offs_n * stride_wn
        w = tl.load(w_ptrs).to(tl.float32)[None, :]
        sigma = tl.rsqrt(tl.sum(y * y, axis=-1, keep_dims=True) / N + eps)
        y = y * sigma * w
        # Fake Quant
        y = _act_fake_quant(y, N, QUANT_BITS, IS_HIP)

    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y.to(y_ptr.type.element_ty), mask=mask)


def fused_silu(
    inputs: torch.Tensor,  # [num_tokens, 2 * hidden_dim]
    expert_cnt: torch.Tensor,  # [num_experts, ]
    norm_weight: torch.Tensor,  # [num_experts, hidden_dim]
    norm_eps: float,
    quant_bits: int,
    max_tokens_per_expert: int,
    num_local_experts: int,
    local_expert_start: int,
):
    num_tokens, hidden_dim = inputs.shape
    hidden_dim //= 2
    outputs = torch.empty((num_tokens, hidden_dim), device=inputs.device, dtype=inputs.dtype)
    if norm_weight is None:
        norm_quant = False
        stride_we = stride_wn = 0
    else:
        assert quant_bits in [2, 4, 8], f"Unsupported bit width: {quant_bits}. Supported values are 2, 4, and 8."
        norm_quant = True
        stride_we = norm_weight.stride(0)
        stride_wn = norm_weight.stride(1)
    block_size_N = triton.next_power_of_2(hidden_dim)
    _fused_silu_kernel[
        lambda META: (
            triton.cdiv(max_tokens_per_expert, META["BLOCK_SIZE_M"]),
            num_local_experts,
        )
    ](
        inputs,
        outputs,
        expert_cnt,
        norm_weight,
        norm_eps,
        inputs.stride(0),
        inputs.stride(1),
        outputs.stride(0),
        outputs.stride(1),
        stride_we,
        stride_wn,
        num_tokens,
        hidden_dim,
        num_local_experts,
        local_expert_start,
        BLOCK_SIZE_N=block_size_N,
        NORM_QUANT=norm_quant,
        QUANT_BITS=quant_bits,
        IS_HIP=is_hip(),
    )
    return outputs


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 1}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_M": 4}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_M": 8}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_M": 16}, num_warps=4, num_stages=1),
    ],
    key=["N"],
)
@triton.jit
def _fused_dsilu_kernel(
    gy_ptr,
    x_ptr,
    gx_ptr,
    y_ptr,
    z_ptr,
    cnt_ptr,
    rmsw_ptr,
    eps,
    stride_gym,
    stride_gyn,
    stride_xm,
    stride_xn,
    stride_gxm,
    stride_gxn,
    stride_ym,
    stride_yn,
    stride_we,
    stride_wn,
    stride_zm,
    stride_zn,
    M,
    N,
    E,
    S,
    NORM_QUANT: tl.constexpr,
    QUANT_BITS: tl.constexpr,
    IS_HIP: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    eid = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=0)

    offset_s = tl.load(cnt_ptr + (S - 1)) if S > 0 else 0
    global_eid = S + eid
    start_m = 0 if global_eid == 0 else tl.load(cnt_ptr + global_eid - 1) - offset_s
    end_m = tl.load(cnt_ptr + global_eid) - offset_s
    if start_m + pid_m * BLOCK_SIZE_M >= end_m:
        return

    offs_m = start_m + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    mask_m = offs_m < end_m
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]

    gy_ptrs = gy_ptr + offs_m[:, None] * stride_gym + offs_n[None, :] * stride_gyn
    x1_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x3_ptrs = x1_ptrs + N * stride_xn
    gy = tl.load(gy_ptrs, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptrs, mask=mask, other=0.0).to(tl.float32)
    x3 = tl.load(x3_ptrs, mask=mask, other=0.0).to(tl.float32)

    s1 = _silu(x1)
    y = s1 * x3

    if NORM_QUANT:
        w_ptrs = rmsw_ptr + eid * stride_we + offs_n * stride_wn
        w = tl.load(w_ptrs).to(tl.float32)[None, :]
        sigma = tl.rsqrt(tl.sum(y * y, axis=-1, keep_dims=True) / N + eps)
        z = y * sigma
        z_ptrs = z_ptr + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn
        tl.store(z_ptrs, (gy * z).to(z_ptr.type.element_ty), mask=mask)
        wgy = w * gy
        gy = sigma * (wgy - z * tl.sum(wgy * z, axis=-1, keep_dims=True) / N)
        y = z * w
        y = _act_fake_quant(y, N, QUANT_BITS, IS_HIP)

    gx3 = gy * s1
    gx1 = gy * x3 * _dsilu(x1)

    gx1_ptrs = gx_ptr + offs_m[:, None] * stride_gxm + offs_n[None, :] * stride_gxn
    gx3_ptrs = gx1_ptrs + N * stride_gxn
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(gx3_ptrs, gx3.to(gx_ptr.type.element_ty), mask=mask)
    tl.store(gx1_ptrs, gx1.to(gx_ptr.type.element_ty), mask=mask)
    tl.store(y_ptrs, y.to(y_ptr.type.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 16}, num_warps=4, num_stages=2),
    ],
    key=["N"],
)
@triton.jit
def _fused_drmsw_kernel(
    gyz_ptr,
    gw_ptr,
    cnt_ptr,
    stride_zm,
    stride_zn,
    stride_we,
    stride_wn,
    M,
    N,
    E,
    S,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    eid = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=0)

    offset_s = tl.load(cnt_ptr + (S - 1)) if S > 0 else 0
    global_eid = S + eid
    start_m = 0 if global_eid == 0 else tl.load(cnt_ptr + global_eid - 1) - offset_s
    end_m = tl.load(cnt_ptr + global_eid) - offset_s

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # TODO: mask_n

    gyz_ptrs = gyz_ptr + (start_m + offs_m[:, None]) * stride_zm + offs_n[None, :] * stride_zn

    acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for m in range(start_m, end_m, BLOCK_SIZE_M):
        mask_m = m + offs_m < end_m
        gyz = tl.load(gyz_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(gyz, axis=0)
        gyz_ptrs += BLOCK_SIZE_M * stride_zm

    gw_ptrs = gw_ptr + eid * stride_we + offs_n * stride_wn
    tl.store(gw_ptrs, acc.to(gw_ptr.type.element_ty))


def fused_dsilu(
    grad_out: torch.Tensor,  # [num_tokens, hidden_dim]
    inputs: torch.Tensor,  # [num_tokens, 2 * hidden_dim]
    expert_cnt: torch.Tensor,  # [num_experts, ]
    norm_weight: torch.Tensor,  # [num_experts, hidden_dim]
    norm_eps: float,
    quant_bits: int,
    max_tokens_per_expert: int,
    num_local_experts: int,
    local_expert_start: int,
):
    num_tokens, hidden_dim = grad_out.shape
    outputs = torch.empty((num_tokens, hidden_dim), device=inputs.device, dtype=inputs.dtype)
    if norm_weight is None:
        norm_quant = False
        gyz = grad_w = None
        stride_we = stride_wn = stride_zm = stride_zn = 0
    else:
        assert quant_bits in [2, 4, 8], f"Unsupported bit width: {quant_bits}. Supported values are 2, 4, and 8."
        norm_quant = True
        gyz = torch.empty((num_tokens, hidden_dim), device=inputs.device, dtype=inputs.dtype)
        grad_w = torch.empty((num_local_experts, hidden_dim), device=inputs.device, dtype=inputs.dtype)
        stride_we = norm_weight.stride(0)
        stride_wn = norm_weight.stride(1)
        stride_zm = gyz.stride(0)
        stride_zn = gyz.stride(1)
    grad_in = torch.empty((num_tokens, 2 * hidden_dim), device=inputs.device, dtype=inputs.dtype)
    outputs = torch.empty((num_tokens, hidden_dim), device=inputs.device, dtype=inputs.dtype)
    block_size_N = triton.next_power_of_2(hidden_dim)
    _fused_dsilu_kernel[
        lambda META: (
            triton.cdiv(max_tokens_per_expert, META["BLOCK_SIZE_M"]),
            num_local_experts,
        )
    ](
        grad_out,
        inputs,
        grad_in,
        outputs,
        gyz,
        expert_cnt,
        norm_weight,
        norm_eps,
        grad_out.stride(0),
        grad_out.stride(1),
        inputs.stride(0),
        inputs.stride(1),
        grad_in.stride(0),
        grad_in.stride(1),
        outputs.stride(0),
        outputs.stride(1),
        stride_we,
        stride_wn,
        stride_zm,
        stride_zn,
        num_tokens,
        hidden_dim,
        num_local_experts,
        local_expert_start,
        BLOCK_SIZE_N=block_size_N,
        NORM_QUANT=norm_quant,
        QUANT_BITS=quant_bits,
        IS_HIP=is_hip(),
    )
    if norm_quant:
        _fused_drmsw_kernel[
            lambda META: (
                triton.cdiv(hidden_dim, META["BLOCK_SIZE_N"]),
                num_local_experts,
            )
        ](
            gyz,
            grad_w,
            expert_cnt,
            stride_zm,
            stride_zn,
            stride_we,
            stride_wn,
            num_tokens,
            hidden_dim,
            num_local_experts,
            local_expert_start,
        )
    return outputs, grad_in, grad_w


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 1}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_M": 4}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_M": 8}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_SIZE_M": 16}, num_warps=4, num_stages=1),
    ],
    key=["N"],
)
@triton.jit
def _scatter_kernel(
    x_ptr,
    y_ptr,
    cnt_ptr,
    idx_ptr,
    rw_ptr,
    stride_xm,
    stride_xn,
    stride_ym,
    stride_yn,
    stride_ie,
    M,
    N,
    E,
    T,
    S,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    WEIGHTED: tl.constexpr,
    QUANT_BITS: tl.constexpr,
    IS_HIP: tl.constexpr,
):
    eid = tl.program_id(axis=1)
    pid_m = tl.program_id(axis=0)

    offset_s = tl.load(cnt_ptr + (S - 1)) if S > 0 else 0
    global_eid = S + eid
    start_m = 0 if global_eid == 0 else tl.load(cnt_ptr + global_eid - 1) - offset_s
    end_m = tl.load(cnt_ptr + global_eid) - offset_s
    if start_m + pid_m * BLOCK_SIZE_M >= end_m:
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    mask = start_m + offs_m < end_m

    idx = tl.load(idx_ptr + global_eid * stride_ie + offs_m, mask=mask, other=0)
    x_idx = idx // T
    y_idx = start_m + offs_m

    offs_n = tl.arange(0, BLOCK_SIZE_N)
    x_ptrs = x_ptr + x_idx[:, None] * stride_xm + offs_n[None, :] * stride_xn
    y_ptrs = y_ptr + y_idx[:, None] * stride_ym + offs_n[None, :] * stride_yn

    data = tl.load(x_ptrs, mask=mask[:, None], other=0.0)

    if WEIGHTED:
        rw = tl.load(rw_ptr + idx, mask=mask, other=0).to(tl.float32)
        data = (rw[:, None] * data).to(y_ptr.type.element_ty)

    if QUANT_BITS > 0:
        data = data.to(tl.float32)
        data = _act_fake_quant(data, N, QUANT_BITS, IS_HIP)
        data = data.to(y_ptr.type.element_ty)

    tl.store(y_ptrs, data, mask=mask[:, None])


def scatter_by_expert(
    inputs: torch.Tensor,  # [num_tokens, hidden_dim]
    expert_cnt: torch.Tensor,  # [num_experts, ]
    expert_idx: torch.Tensor,  # [num_experts, num_tokens]
    routing_weights: torch.Tensor,  # [num_tokens, top_k]
    max_tokens_per_expert: int,
    num_local_tokens: int,
    num_local_experts: int,
    local_expert_start: int,
    quant_bits: int = 0,
    weighted: bool = False,
):
    num_tokens, hidden_dim = inputs.shape
    top_k = routing_weights.shape[-1]
    outputs = torch.empty(
        size=(num_local_tokens, hidden_dim),
        device=inputs.device,
        dtype=inputs.dtype,
    )
    block_size_N = triton.next_power_of_2(hidden_dim)
    _scatter_kernel[
        lambda META: (
            triton.cdiv(max_tokens_per_expert, META["BLOCK_SIZE_M"]),
            num_local_experts,
        )
    ](
        inputs,
        outputs,
        expert_cnt,
        expert_idx,
        routing_weights,
        inputs.stride(0),
        inputs.stride(1),
        outputs.stride(0),
        outputs.stride(1),
        expert_idx.stride(0),
        num_tokens,
        hidden_dim,
        num_local_experts,
        top_k,
        local_expert_start,
        WEIGHTED=weighted,
        QUANT_BITS=quant_bits,
        BLOCK_SIZE_N=block_size_N,
        IS_HIP=is_hip(),
    )
    return outputs


@triton.jit
def _merge_kernel(
    x_ptr,
    i_ptr,
    w_ptr,
    y_ptr,
    stride_xm,
    stride_xt,
    stride_xn,
    stride_im,
    stride_it,
    stride_wm,
    stride_wt,
    stride_ym,
    stride_yn,
    M,
    N,
    T,
    S,
    E,
    BLOCK_SIZE_N: tl.constexpr,
    WEIGHTED: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    mask = offs_n < N
    x_ptrs = x_ptr + pid_m * stride_xm + offs_n * stride_xn
    w_ptrs = w_ptr + pid_m * stride_wm
    i_ptrs = i_ptr + pid_m * stride_im
    y_ptrs = y_ptr + pid_m * stride_ym + offs_n * stride_yn
    acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    local_expert_start = S
    local_expert_end = S + E
    for t in range(T):
        i = tl.load(i_ptrs)
        if i >= local_expert_start and i < local_expert_end:
            x = tl.load(x_ptrs)
            if WEIGHTED:
                w = tl.load(w_ptrs)
                acc += x * w
            else:
                acc += x
        x_ptrs += stride_xt
        i_ptrs += stride_it
        w_ptrs += stride_wt
    acc = acc.to(y_ptr.type.element_ty)
    tl.store(y_ptrs, acc, mask=mask)


def merge_topk(
    inputs: torch.Tensor,
    expert_index: torch.Tensor,
    expert_weight: torch.Tensor,
    local_expert_start: int,
    num_local_experts: int,
    weighted: bool = False,
):
    num_tokens, top_k, hidden_dim = inputs.shape
    outputs = torch.zeros((num_tokens, hidden_dim), dtype=inputs.dtype, device=inputs.device)
    _merge_kernel[(num_tokens, 1, 1)](
        inputs,
        expert_index,
        expert_weight,
        outputs,
        inputs.stride(0),
        inputs.stride(1),
        inputs.stride(2),
        expert_index.stride(0),
        expert_index.stride(1),
        expert_weight.stride(0),
        expert_weight.stride(1),
        outputs.stride(0),
        outputs.stride(1),
        num_tokens,
        hidden_dim,
        top_k,
        local_expert_start,
        num_local_experts,
        BLOCK_SIZE_N=triton.next_power_of_2(hidden_dim),
        WEIGHTED=weighted,
        num_warps=8,
        num_stages=1,
    )
    return outputs


@torch.compile
def compute_grw(gy: torch.Tensor, y2: torch.Tensor):
    return torch.sum(gy[:, None, :] * y2, dim=-1)


@torch.compile
def weight_quant_reduce(x: torch.Tensor):
    return 1.0 / x.float().abs().mean(dim=-1, keepdim=True).clamp_(min=1e-5)


@torch.compile
def weight_quant_scale(x: torch.Tensor, s: torch.Tensor):
    return ((x * s).round().clamp(-1, 1) / s).to(x.dtype)


def weight_quant(x: torch.Tensor):
    s = weight_quant_reduce(x)
    return weight_quant_scale(x, s)


class GroupGEMM(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,  # [batch_size * sequence_length, hidden_dim]
        w: torch.Tensor,  # [num_experts, hidden_dim, hidden_dim]
        cnt: torch.Tensor,  # [num_experts, ]
    ):
        num_tokens, hidden_dim = x.shape
        num_experts = cnt.shape[0]
        ffn_dim = w.shape[1]
        max_tokens_per_expert = cnt.max().item()

        TIMING_LOGGER("[fwd] cumsum")
        cnt = torch.cumsum(cnt, dim=0, dtype=torch.int32)
        TIMING_LOGGER("[fwd] cumsum")

        TIMING_LOGGER("[fwd] gemm")
        y = grouped_gemm_sparse_M(
            x,
            w,
            cnt,
            None,
            num_tokens,
            ffn_dim,
            hidden_dim,
            num_experts,
            -1,
            0,
            max_tokens_per_expert,
            forks_A=1,
            sparse_A=False,
            sparse_C=False,
            transpose_B=True,
        )
        TIMING_LOGGER("[fwd] w13")

        ctx.save_for_backward(x, w, cnt)
        ctx.num_tokens = num_tokens
        ctx.hidden_dim = hidden_dim
        ctx.num_experts = num_experts
        ctx.ffn_dim = ffn_dim
        ctx.max_tokens_per_expert = max_tokens_per_expert
        return y

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_y: torch.Tensor):
        x, w, cnt = ctx.saved_tensors

        TIMING_LOGGER("[bwd] dw")
        grad_w = grouped_gemm_sparse_K(
            grad_y,
            x,
            cnt,
            None,
            ctx.ffn_dim,
            ctx.hidden_dim,
            ctx.num_tokens,
            ctx.num_experts,
            -1,
            0,
            sparse_A=False,
            sparse_B=False,
            forks_A=1,
            forks_B=1,
        )
        TIMING_LOGGER("[bwd] dw")

        TIMING_LOGGER("[bwd] dx")
        grad_x = grouped_gemm_sparse_M(
            grad_y,
            w,
            cnt,
            None,
            ctx.num_tokens,
            ctx.hidden_dim,
            ctx.ffn_dim,
            ctx.num_experts,
            -1,
            0,
            ctx.max_tokens_per_expert,
            forks_A=1,
            sparse_A=False,
            sparse_C=False,
            transpose_B=False,
        )
        TIMING_LOGGER("[bwd] dx")

        return grad_x, grad_w, None


def gmm(x: torch.Tensor, w: torch.Tensor, cnt: torch.Tensor):
    return GroupGEMM.apply(x, w, cnt)


class MoEFFNFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,  # [batch_size * sequence_length, hidden_dim]
        w13: torch.Tensor,  # [num_experts, 2 * ffn_dim, hidden_dim]
        w2: torch.Tensor,  # [num_experts, hidden_dim, ffn_dim]
        cnt: torch.Tensor,  # [num_experts, ]
        idx: torch.Tensor,  # [num_experts, batch_size * sequence_length]
        ri: torch.Tensor,  # [batch_size * sequence_length, top_k]
        rw: torch.Tensor,  # [batch_size * sequence_length, top_k]
        rmsw: torch.Tensor = None,  # [num_experts, ffn_dim]
        eps: float = None,
        tp_group: dist.ProcessGroup = None,
        ep_group: dist.ProcessGroup = None,
    ):
        if dist.is_initialized() and ep_group is not None:
            ep_rank, ep_size = dist.get_rank(ep_group), dist.get_world_size(ep_group)
        else:
            ep_rank, ep_size = 0, 1

        num_tokens, hidden_dim = x.shape
        ffn_dim = w2.shape[-1]
        num_experts = cnt.shape[0]
        top_k = rw.shape[-1]

        num_local_experts = num_experts // ep_size
        local_expert_start = ep_rank * num_local_experts
        local_expert_end = local_expert_start + num_local_experts
        max_tokens_per_expert = cnt[local_expert_start:local_expert_end].max().item()

        if rmsw is not None:
            act_quant_bits = 8
            # weight_quant_bits = "1.58"
        else:
            act_quant_bits = 0
            # weight_quant_bits = "0"

        TIMING_LOGGER("[fwd] cumsum")
        cnt = torch.cumsum(cnt, dim=0, dtype=torch.int32)
        local_token_start = cnt[local_expert_start - 1].item() if local_expert_start > 0 else 0
        local_token_end = cnt[local_expert_end - 1].item()
        num_local_tokens = local_token_end - local_token_start
        TIMING_LOGGER("[fwd] cumsum")

        TIMING_LOGGER("[fwd] scatter")
        x13 = scatter_by_expert(
            x,
            cnt,
            idx,
            rw,
            max_tokens_per_expert,
            num_local_tokens,
            num_local_experts,
            local_expert_start,
            weighted=False,
            quant_bits=act_quant_bits,
        )
        TIMING_LOGGER("[fwd] scatter")
        w13_sparse_A = False
        w13_forks_A = 1

        # BitLinear: quant w13
        if rmsw is not None:
            TIMING_LOGGER("[fwd] qw13")
            w13 = weight_quant(w13.view((num_experts * 2, -1))).view(w13.shape)
            TIMING_LOGGER("[fwd] qw13")

        # Y13 <= X13 @ W13.T
        TIMING_LOGGER("[fwd] w13")
        y13 = grouped_gemm_sparse_M(
            x13,
            w13,
            cnt,
            idx,
            num_local_tokens,
            2 * ffn_dim,
            hidden_dim,
            num_local_experts,
            top_k,
            local_expert_start,
            max_tokens_per_expert,
            forks_A=w13_forks_A,
            sparse_A=w13_sparse_A,
            sparse_C=False,
            transpose_B=True,
        )
        TIMING_LOGGER("[fwd] w13")

        # X2 <= silu(Y1) * Y3
        TIMING_LOGGER("[fwd] silu")
        x2 = fused_silu(
            y13,
            cnt,
            rmsw,
            eps,
            quant_bits=act_quant_bits,
            max_tokens_per_expert=max_tokens_per_expert,
            num_local_experts=num_local_experts,
            local_expert_start=local_expert_start,
        )
        TIMING_LOGGER("[fwd] silu")

        # BitLinear: quant w2
        if rmsw is not None:
            TIMING_LOGGER("[fwd] qw2")
            w2 = weight_quant(w2.view((num_experts, -1))).view(w2.shape)
            TIMING_LOGGER("[fwd] qw2")

        # Y <= X2 @ W2.T
        TIMING_LOGGER("[fwd] w2")
        y2 = grouped_gemm_sparse_M(
            x2,
            w2,
            cnt,
            idx,
            num_tokens,
            hidden_dim,
            ffn_dim,
            num_local_experts,
            top_k,
            local_expert_start,
            max_tokens_per_expert,
            forks_A=1,
            sparse_A=False,
            sparse_C=True,
            transpose_B=True,
        )
        TIMING_LOGGER("[fwd] w2")

        # Y <= sum(RW * Y)
        TIMING_LOGGER("[fwd] merge")
        y2 = y2.view((num_tokens, top_k, hidden_dim))
        y = merge_topk(y2, ri, rw, local_expert_start, num_local_experts, weighted=True)
        TIMING_LOGGER("[fwd] merge")

        if dist.is_initialized():
            TIMING_LOGGER("[fwd] reduce")
            if tp_group is not None and dist.get_world_size(tp_group) > 1:
                dist.all_reduce(y, op=dist.ReduceOp.SUM, group=tp_group)
            if ep_group is not None and dist.get_world_size(ep_group) > 1:
                dist.all_reduce(y, op=dist.ReduceOp.SUM, group=ep_group)
            TIMING_LOGGER("[fwd] reduce")

        # TODO: check partition plan to save x or x13
        ctx.save_for_backward(x13, w13, w2, cnt, idx, ri, rw, y13, y2, rmsw)
        ctx.tp_group = tp_group
        ctx.ep_group = ep_group
        ctx.num_tokens = num_tokens
        ctx.hidden_dim = hidden_dim
        ctx.ffn_dim = ffn_dim
        ctx.num_experts = num_experts
        ctx.top_k = top_k
        ctx.max_tokens_per_expert = max_tokens_per_expert
        ctx.num_local_tokens = num_local_tokens
        ctx.num_local_experts = num_local_experts
        ctx.local_expert_start = local_expert_start
        ctx.act_quant_bits = act_quant_bits
        ctx.eps = eps
        return y

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        gy: torch.Tensor,  # [batch_size * sequence_length, hidden_dim]
    ):
        x13, w13, w2, cnt, idx, ri, rw, y13, y2, rmsw = ctx.saved_tensors

        # gRW <= sum(Y * gY0) TODO: Fuse
        TIMING_LOGGER("[bwd] grw")
        grw = compute_grw(gy, y2)
        if dist.is_initialized():
            if ctx.tp_group is not None and dist.get_world_size(ctx.tp_group) > 1:
                dist.all_reduce(grw, op=dist.ReduceOp.SUM, group=ctx.tp_group)
            if ctx.ep_group is not None and dist.get_world_size(ctx.ep_group) > 1:
                dist.all_reduce(grw, op=dist.ReduceOp.SUM, group=ctx.ep_group)
        TIMING_LOGGER("[bwd] grw")

        TIMING_LOGGER("[bwd] scatter")
        gy = scatter_by_expert(
            gy,
            cnt,
            idx,
            rw,
            ctx.max_tokens_per_expert,
            ctx.num_local_tokens,
            ctx.num_local_experts,
            ctx.local_expert_start,
            weighted=True,
        )
        TIMING_LOGGER("[bwd] scatter")

        # gX2 <= gY @ W2
        TIMING_LOGGER("[bwd] gx2")
        gx2 = grouped_gemm_sparse_M(
            gy,
            w2,
            cnt,
            idx,
            ctx.num_local_tokens,
            ctx.ffn_dim,
            ctx.hidden_dim,
            ctx.num_local_experts,
            ctx.top_k,
            ctx.local_expert_start,
            ctx.max_tokens_per_expert,
            forks_A=1,
            sparse_A=False,
            sparse_C=False,
            transpose_B=False,
        )
        TIMING_LOGGER("[bwd] gx2")

        # gY1 <= gX2 * Y3 * dsilu(Y1), gY3 <= gX2 * silu(Y1)
        TIMING_LOGGER("[bwd] dsilu")
        x2, gy13, grmsw = fused_dsilu(
            gx2,
            y13,
            cnt,
            rmsw,
            ctx.eps,
            quant_bits=ctx.act_quant_bits,
            max_tokens_per_expert=ctx.max_tokens_per_expert,
            num_local_experts=ctx.num_local_experts,
            local_expert_start=ctx.local_expert_start,
        )
        TIMING_LOGGER("[bwd] dsilu")

        # gW2 <= gY.T @ X2
        TIMING_LOGGER("[bwd] gw2")
        gw2 = grouped_gemm_sparse_K(
            gy,
            x2,
            cnt,
            idx,
            ctx.hidden_dim,
            ctx.ffn_dim,
            ctx.num_local_tokens,
            ctx.num_local_experts,
            ctx.top_k,
            ctx.local_expert_start,
            sparse_A=False,
            sparse_B=False,
            forks_A=1,
            forks_B=1,
        )
        TIMING_LOGGER("[bwd] gw2")

        # gW13 <= gY13.T @ X13
        TIMING_LOGGER("[bwd] gw13")
        gw13 = grouped_gemm_sparse_K(
            gy13,
            x13,
            cnt,
            idx,
            2 * ctx.ffn_dim,
            ctx.hidden_dim,
            ctx.num_tokens,
            ctx.num_local_experts,
            ctx.top_k,
            ctx.local_expert_start,
            sparse_A=False,
            sparse_B=False,
            forks_A=1,
            forks_B=1,
        )
        TIMING_LOGGER("[bwd] gw13")

        # gX13 <= gY13 @ W13
        TIMING_LOGGER("[bwd] gx13")
        gx13 = grouped_gemm_sparse_M(
            gy13,
            w13,
            cnt,
            idx,
            ctx.num_tokens,
            ctx.hidden_dim,
            2 * ctx.ffn_dim,
            ctx.num_local_experts,
            ctx.top_k,
            ctx.local_expert_start,
            ctx.max_tokens_per_expert,
            forks_A=1,
            sparse_A=False,
            sparse_C=True,
            transpose_B=False,
        )
        TIMING_LOGGER("[bwd] gx13")

        # gX <= sum(gX13)
        TIMING_LOGGER("[bwd] merge")
        gx = merge_topk(
            gx13.view((ctx.num_tokens, ctx.top_k, ctx.hidden_dim)),
            ri,
            rw,
            ctx.local_expert_start,
            ctx.num_local_experts,
            weighted=False,
        )
        TIMING_LOGGER("[bwd] merge")

        if dist.is_initialized():
            TIMING_LOGGER("[bwd] reduce")
            if ctx.tp_group is not None and dist.get_world_size(ctx.tp_group) > 1:
                dist.all_reduce(gx, op=dist.ReduceOp.SUM, group=ctx.tp_group)
            if ctx.ep_group is not None and dist.get_world_size(ctx.ep_group) > 1:
                dist.all_reduce(gx, op=dist.ReduceOp.SUM, group=ctx.ep_group)
            TIMING_LOGGER("[bwd] reduce")

        return gx, gw13, gw2, None, None, None, grw, grmsw, None, None, None


def fused_moe_ffn(
    hidden_states: torch.Tensor,  # [batch_size * sequence_length, hidden_dim]
    w13: torch.Tensor,  # [num_experts, 2 * ffn_dim, hidden_dim]
    w2: torch.Tensor,  # [num_experts, hidden_dim, ffn_dim]
    expert_cnt: torch.Tensor,  # [num_experts, ]
    expert_idx: torch.Tensor,  # [num_experts, batch_size * sequence_length]
    selected_experts: torch.Tensor,  # [batch_size * sequence_length, top_k]
    routing_weights: torch.Tensor,  # [batch_size * sequence_length, top_k]
    rmsw: torch.Tensor = None,  # [num_experts, ffn_dim]
    eps: float = None,
    tp_group: dist.ProcessGroup = None,
    ep_group: dist.ProcessGroup = None,
):
    return MoEFFNFunction.apply(
        hidden_states,
        w13,
        w2,
        expert_cnt,
        expert_idx,
        selected_experts,
        routing_weights,
        rmsw,
        eps,
        tp_group,
        ep_group,
    )


def nnscaler_fused_moe_ffn(
    hidden_states: torch.Tensor,  # [batch_size * sequence_length, hidden_dim]
    selected_experts: torch.Tensor,  # [batch_size * sequence_length, top_k]
    routing_weights: torch.Tensor,  # [batch_size * sequence_length, top_k]
    w13: torch.Tensor,  # [num_experts, 2 * ffn_dim, hidden_dim]
    w2: torch.Tensor,  # [num_experts, hidden_dim, ffn_dim]
    expert_cnt: torch.Tensor,  # [num_experts, ]
    expert_idx: torch.Tensor,  # [num_experts, batch_size * sequence_length]
    global_expert_num: int,  # a var used to generate data for profiling, not used in runtime
):
    return MoEFFNFunction.apply(hidden_states, w13, w2, expert_cnt, expert_idx, selected_experts, routing_weights)
