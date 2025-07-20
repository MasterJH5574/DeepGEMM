import torch
from typing import List, Tuple, Optional, Dict

from ..jit import build
from .runtime import Runtime, FP8WGradGemmRuntime
from .gemm import get_best_configs
from .utils import ceil_div, get_num_sms

# @torch.compiler.disable
def wgrad_gemm_fp8_fp8_fp32_nt(lhs: Tuple[torch.Tensor, torch.Tensor],
                               rhs: Tuple[torch.Tensor, torch.Tensor],
                               runtime_cache: Optional[Dict[str, Runtime]] = None) -> torch.Tensor:
    """
    Perform a weight gradient GEMM with FP8 inputs and FP32 output, with 1x128 LHS scaling and 1x128 RHS scaling.
        Results will be accumulated into the output tensor.

    Requirements:
        LHS, RHS, and output tensors must be contiguous in dimension 1, i.e., stride(1) = 1.
        The stride(0) of LHS and RHS must be a multiple of 16, and the stride(0) of output must be a multiple of 4.
        RHS and RHS scaling factors are required to be transposed.
        The LHS scaling and RHS scaling tensor require a TMA-aligned transposed format.
            If your input does not match the requirement, this function will do a transposing with a set of slow PyTorch operations.

    Arguments:
        lhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[m, k]`,
             the second element is an FP32 1x128 scaling tensor for LHS of shape `[m, ⌈k / 128⌉]`.
        rhs: the first element is an FP8 tensor (typed `torch.float8_e4m3fn`) of shape `[n, k]`,
             the second element is an FP32 1x128 scaling tensor for RHS of shape `[n, ⌈k / 128⌉]`.
        out: the FP32 output tensor of shape `[m, n]`, which will be accumulated.
    """
    lhs, lhs_scales = lhs
    rhs, rhs_scales = rhs
    m, k = lhs.shape
    n, _ = rhs.shape
    aligned_k = ceil_div(k, 128) * 128
    # NOTE: keep the comments below for k-grouped GEMMs
    # assert lhs_scales.shape == (m, ceil_div(k, 128)) or lhs_scales.shape == (ceil_div(k, 128), m)
    # assert rhs_scales.shape == (n, ceil_div(k, 128)) or rhs_scales.shape == (ceil_div(k, 128), n)
    # LHS and RHS scales must be transposed for TMA load
    # NOTES: `get_col_major_tma_aligned_tensor` may launch a kernel if not processed by previous kernels
    # def get_valid_scales(scales: torch.Tensor, mn: int):
    #     if scales.shape == (ceil_div(k, 128), mn):
    #         # For k-grouped GEMMs
    #         scales = scales.permute(1, 0)
    #         assert get_tma_aligned_size(mn, 4) == scales.stride(1) == mn
    #     return scales
    # lhs_scales = get_valid_scales(lhs_scales, m)
    # rhs_scales = get_valid_scales(rhs_scales, n)

    # Auto-tuning with compilation
    num_sms = get_num_sms()
    num_sms, block_m, block_n, num_stages, tma_multicast_config, smem_config = get_best_configs(
        m, n, aligned_k, 1, num_sms, is_fp32_out=True, is_wgrad=True)
    num_last_stages = ceil_div(k, 128) % num_stages
    block_k = 128
    num_tma_threads = 128
    num_math_threads_per_group = 128

    # Generate, build and run the kernel
    code = FP8WGradGemmRuntime.generate(m, n, block_m, block_n, block_k, num_stages, num_last_stages, num_tma_threads, num_math_threads_per_group, tma_multicast_config[0], tma_multicast_config[1])
    if runtime_cache is not None and code in runtime_cache:
        runtime = runtime_cache[code]
    else:
        runtime = build('wgrad_gemm_fp8_fp8_fp32_nt', code, FP8WGradGemmRuntime)
        if runtime_cache is not None:
            runtime_cache[code] = runtime
    return runtime(lhs, rhs, lhs_scales, rhs_scales, block_m, block_n, block_k, 1, smem_config[1], num_sms, tma_multicast_config[0], smem_config[0])


def k_grouped_wgrad_gemm_fp8_fp8_fp32_nt(lhs: Tuple[torch.Tensor, torch.Tensor],
                                         rhs: Tuple[torch.Tensor, torch.Tensor],
                                         out: torch.Tensor,
                                         batch_sizes: List[int]):
    """
    Perform a k-grouped weight gradient GEMM with FP8 inputs and FP32 output, with 1x128 LHS scaling and 1x128 RHS scaling.
        Results will be accumulated into the output tensor.

    Requirements:
        This function handles multiple batches with varying k-dimensions, processing each batch sequentially.
        Each batch's LHS, RHS, and output tensors must be contiguous.
        The RHS and RHS scaling factors are required to be transposed.
        The LHS scaling and RHS scaling tensors require a TMA-aligned transposed format.

    Arguments:
        lhs: The first element is a flattened FP8 tensor (typed `torch.float8_e4m3fn`) containing all batches of LHS data,
                 and the flattened shape is `[sum(m * k for k in batch_sizes)]`, where m is the number of rows.
             The second element is an FP32 scaling tensor for LHS with shape `[⌈k / 128⌉ for k in batch_sizes), m]`,
                 representing the per-128-channel scaling factors.
        rhs: The first element is a flattened FP8 tensor (typed `torch.float8_e4m3fn`) containing all batches of RHS data,
                 and the flattened shape is `[sum(n * k for k in batch_sizes)]`, where n is the number of rows.
             The second element is an FP32 scaling tensor for RHS with shape `[⌈k / 128⌉ for k in batch_sizes), n]`,
                 representing the per-128-channel scaling factors.
        out: The FP32 output tensor of shape [num_batches, m, n], which will be accumulated.
        batch_sizes: A list of integers specifying the k-dimension for each batch.
    """
    lhs, lhs_scales = lhs[0].view(-1), lhs[1]
    rhs, rhs_scales = rhs[0].view(-1), rhs[1]
    num_batches, m, n = out.shape

    lhs_offset, rhs_offset, scales_offset = 0, 0, 0

    for i in range(num_batches):
        k = batch_sizes[i]
        lhs_slice = lhs[lhs_offset:lhs_offset + m * k].view(m, k)
        rhs_slice = rhs[rhs_offset:rhs_offset + n * k].view(n, k)
        lhs_scales_slice = lhs_scales[scales_offset:scales_offset + ceil_div(k, 128)]
        rhs_scales_slice = rhs_scales[scales_offset:scales_offset + ceil_div(k, 128)]
        wgrad_gemm_fp8_fp8_fp32_nt((lhs_slice, lhs_scales_slice), (rhs_slice, rhs_scales_slice), out[i])

        lhs_offset += m * k
        rhs_offset += n * k
        scales_offset += ceil_div(k, 128)
