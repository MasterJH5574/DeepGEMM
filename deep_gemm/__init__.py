import torch

from . import jit
from .jit_kernels import (
    gemm_fp8_fp8_bf16_nt,
    m_grouped_gemm_fp8_fp8_bf16_nt_contiguous,
    m_grouped_gemm_fp8_fp8_bf16_nt_masked,
    wgrad_gemm_fp8_fp8_fp32_nt,
    k_grouped_wgrad_gemm_fp8_fp8_fp32_nt,
    ceil_div,
    set_num_sms, get_num_sms,
    get_col_major_tma_aligned_tensor,
    get_m_alignment_for_contiguous_layout
)
from .utils import bench, bench_kineto, calc_diff

import deepgemm_runtime

torch.ops.load_library(deepgemm_runtime.__file__)

# fmt: off
@torch.library.register_fake("deepgemm_runtime::launch_fp8_gemm")
def _(kernel_handle, a, b, scales_a, scales_b, block_m, block_n, block_k, num_groups, d_swizzle_mode, num_sms, num_tma_multicast, smem_size) -> None:
    m, _ = a.shape
    n, _ = b.shape
    return torch.empty((m, n), device=a.device, dtype=torch.bfloat16)

@torch.library.register_fake("deepgemm_runtime::launch_wgrad_fp8_gemm")
def _(kernel_handle, a, b, scales_a, scales_b, block_m, block_n, block_k, num_groups, d_swizzle_mode, num_sms, num_tma_multicast, smem_size) -> None:
    m, _ = a.shape
    n, _ = b.shape
    return torch.zeros((m, n), device=a.device, dtype=torch.float32)
# fmt: on
