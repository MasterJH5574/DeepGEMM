import ctypes
import enum
import os
from typing import Any, Dict, Tuple

import cuda.bindings.driver as cbd
from torch.ops import deepgemm_runtime
import torch

from ..jit.runtime import Runtime
from .utils import get_tma_aligned_size


class GemmType(enum.Enum):
    Normal = 0
    GroupedContiguous = 1
    GroupedMasked = 2

    def __str__(self) -> str:
        return {
            0: 'Normal',
            1: 'GroupedContiguous',
            2: 'GroupedMasked',
        }[self.value]


tmap_type_map: Dict[Any, str] = {
    torch.int8:            cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.int16:           cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT16,
    torch.int32:           cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT32,
    torch.int64:           cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_INT64,
    torch.uint8:           cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.uint16:          cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT16,
    torch.uint32:          cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT32,
    torch.uint64:          cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT64,
    torch.float32:         cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
    torch.float16:         cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    torch.bfloat16:        cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
    torch.float8_e4m3fn:   cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.float8_e4m3fnuz: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.float8_e5m2:     cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    torch.float8_e5m2fnuz: cbd.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
}

swizzle_type_map = {
    0:   cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE,
    32:  cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
    64:  cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B,
    128: cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
}


def get_num_math_warpgroups(block_m: int) -> int:
    return 1 if block_m == 64 else 2


def get_num_threads_per_sm(num_tma_threads: int, num_math_threads_per_group: int, block_m: int) -> int:
    assert num_math_threads_per_group == 128, 'Only support 128 threads per math group'
    return get_num_math_warpgroups(block_m) * num_math_threads_per_group + num_tma_threads


def make_2d_tma_copy_desc(t: torch.Tensor,
                          gmem_dims: Tuple[cbd.cuuint64_t, cbd.cuuint64_t], gmem_outer_stride: cbd.cuuint64_t,
                          smem_dims: Tuple[cbd.cuuint32_t, cbd.cuuint32_t],
                          swizzle_type: cbd.CUtensorMapSwizzle) -> cbd.CUtensorMap:
    tensor_dtype = tmap_type_map[t.dtype]
    # print(f"make_2d_tma_desc, gmem_dims: {gmem_dims}, gmem_outer_stride: {gmem_outer_stride}, smem_dims: {smem_dims}, swizzle_type: {swizzle_type}")
    res, tensor_map = cbd.cuTensorMapEncodeTiled(
        tensor_dtype,
        2,
        t.data_ptr(),
        gmem_dims,
        (gmem_outer_stride,),
        smem_dims,
        (cbd.cuuint32_t(1), cbd.cuuint32_t(1)),
        cbd.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle_type,
        cbd.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        cbd.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )

    if res != cbd.CUresult.CUDA_SUCCESS:
        raise Exception(f'Failed to encode tensor map: {res}')
    return tensor_map


def make_2d_tma_desc(t: torch.Tensor,
                     gmem_inner_dim: int, gmem_outer_dim: int, gmem_outer_stride: int,
                     smem_inner_dim: int, smem_outer_dim: int,
                     swizzle_type: cbd.CUtensorMapSwizzle = cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B) -> cbd.CUtensorMap:
    gmem_dim = (cbd.cuuint64_t(gmem_inner_dim), cbd.cuuint64_t(gmem_outer_dim))
    smem_dim = (cbd.cuuint32_t(smem_inner_dim), cbd.cuuint32_t(smem_outer_dim))
    return make_2d_tma_copy_desc(t, gmem_dim, cbd.cuuint64_t(gmem_outer_stride * t.element_size()), smem_dim, swizzle_type)


def make_2d_tma_a_desc(gemm_type: GemmType, t: torch.Tensor,
                       shape_m: int, shape_k: int, m_stride: int,
                       block_m: int, block_k: int,
                       num_groups: int) -> cbd.CUtensorMap:
    return make_2d_tma_desc(t,
                            shape_k, shape_m * (num_groups if gemm_type == GemmType.GroupedMasked else 1), m_stride,
                            block_k, block_m)


def make_2d_tma_b_desc(gemm_type: GemmType, t: torch.Tensor,
                       shape_n: int, shape_k: int, n_stride: int,
                       block_n: int, block_k: int,
                       num_groups: int) -> cbd.CUtensorMap:
    return make_2d_tma_desc(t,
                            shape_k, shape_n * (num_groups if gemm_type != GemmType.Normal else 1), n_stride,
                            block_k, block_n)


def make_2d_tma_d_desc(gemm_type: GemmType, t: torch.Tensor,
                       shape_m: int, shape_n: int, m_stride: int,
                       block_m: int, block_n: int,
                       num_groups: int,
                       swizzle_mode: int) -> cbd.CUtensorMap:
    # Swizzling requires the inner box dim to be less or equal than `kSwizzleDMode`
    # bytes, so `BLOCK_N * sizeof(T) / kSwizzleDMode` TMA stores are required
    return make_2d_tma_desc(t,
                            shape_n, shape_m * (num_groups if gemm_type == GemmType.GroupedMasked else 1), m_stride,
                            block_n if swizzle_mode == 0 else swizzle_mode // t.element_size(), block_m,
                            swizzle_type_map[swizzle_mode])


def make_2d_tma_scales_desc(gemm_type: GemmType, t: torch.Tensor,
                            shape_mn: int, shape_k: int,
                            block_mn: int, block_k: int,
                            num_groups: int) -> cbd.CUtensorMap:
    # Make TMA aligned to 16 bytes
    shape_mn = get_tma_aligned_size(shape_mn, t.element_size())
    return make_2d_tma_desc(t,
                            shape_mn, (shape_k + block_k - 1) // block_k * (num_groups if gemm_type == GemmType.GroupedMasked else 1), shape_mn,
                            block_mn, 1,
                            cbd.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE)


class FP8GemmRuntime(Runtime):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    @staticmethod
    def generate(N, K, BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_N_PADDING, SWIZZLE_D_MODE, NUM_GROUPS, NUM_STAGES, NUM_TMA_THREADS, NUM_MATH_THREADS_PER_GROUP, NUM_TMA_MULTICAST, IS_TMA_MULTICAST_ON_A, GEMM_TYPE) -> str:
        code = f'''
#ifdef __CUDACC_RTC__
#include <deep_gemm/nvrtc_std.cuh>
#else
#include <cuda.h>
#include <string>
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <deep_gemm/fp8_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&fp8_gemm_kernel<
        {N},
        {K},
        {BLOCK_M},
        {BLOCK_N},
        {BLOCK_K},
        {BLOCK_N_PADDING},
        {SWIZZLE_D_MODE},
        {NUM_GROUPS},
        {NUM_STAGES},
        {NUM_TMA_THREADS},
        {NUM_MATH_THREADS_PER_GROUP},
        {NUM_TMA_MULTICAST},
        {'true' if IS_TMA_MULTICAST_ON_A else 'false'},
        GemmType::{GEMM_TYPE}
      >);
}};
'''
        # if int(os.getenv('DG_JIT_DEBUG', 0)):
        #     print(f'Generated FP8 GEMM code:\n{code}')
        return code

    # noinspection PyMethodOverriding
    @staticmethod
    def launch(kernel, args) -> cbd.CUresult:
        return deepgemm_runtime.launch_fp8_gemm(kernel, *args)


class FP8WGradGemmRuntime(Runtime):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    @staticmethod
    def generate(M, N, BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, NUM_LAST_STAGES, NUM_TMA_THREADS, NUM_MATH_THREADS_PER_GROUP, NUM_TMA_MULTICAST, IS_TMA_MULTICAST_ON_A) -> str:
        code = f'''
#ifdef __CUDACC_RTC__
#include <deep_gemm/nvrtc_std.cuh>
#else
#include <cuda.h>
#include <string>
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <deep_gemm/fp8_wgrad_gemm.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&fp8_wgrad_gemm_kernel<
        {M},
        {N},
        {BLOCK_M},
        {BLOCK_N},
        {BLOCK_K},
        {NUM_STAGES},
        {NUM_LAST_STAGES},
        {NUM_TMA_THREADS},
        {NUM_MATH_THREADS_PER_GROUP},
        {NUM_TMA_MULTICAST},
        {'true' if IS_TMA_MULTICAST_ON_A else 'false'}
      >);
}};
'''
        # if int(os.getenv('DG_JIT_DEBUG', 0)):
        #     print(f'Generated FP8 WGrad GEMM code:\n{code}')
        return code

    # noinspection PyMethodOverriding
    @staticmethod
    def launch(kernel, args) -> cbd.CUresult:
        return deepgemm_runtime.launch_wgrad_fp8_gemm(kernel, *args)
