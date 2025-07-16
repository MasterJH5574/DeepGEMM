#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <torch/all.h>
#include <torch/extension.h>
#include <torch/library.h>

namespace lotus_deepgemm {

inline std::string GetCUDAErrorName(CUresult result) {
  const char* error_name = nullptr;
  cuGetErrorName(result, &error_name);
  return std::string(error_name);
}

inline uint32_t GetTMAAlignedSize(uint32_t x, uint32_t element_size) {
  uint32_t tma_alignment_bytes = 16;
  TORCH_CHECK(tma_alignment_bytes % element_size == 0,
              "TMA alignment bytes must be divisible by element size");
  uint32_t alignment = tma_alignment_bytes / element_size;
  return (x + alignment - 1) / alignment * alignment;
}

inline int GetNumMathWarpGroups(int block_m) { return block_m == 64 ? 1 : 2; }

inline int GetNumThreadsPerSM(int num_tma_threads, int num_math_threads_per_group, int block_m) {
  TORCH_CHECK(num_math_threads_per_group == 128, "Only support 128 threads per math group");
  return GetNumMathWarpGroups(block_m) * num_math_threads_per_group + num_tma_threads;
}

enum GemmType : int {
  kNormal = 0,
  kGroupedContiguous = 1,
  kGroupedMasked = 2,
};

CUtensorMap MakeSingle2DTMADesc(torch::Tensor t, cuuint64_t gmem_inner_dim,
                                cuuint64_t gmem_outer_dim, cuuint64_t gmem_outer_stride,
                                cuuint32_t smem_inner_dim, cuuint32_t smem_outer_dim,
                                CUtensorMapSwizzle swizzle_type) {
  CUtensorMapDataType tensor_dtype;
  if (t.dtype() == torch::kFloat8_e4m3fn) {
    tensor_dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else if (t.dtype() == torch::kFloat32) {
    tensor_dtype = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  } else if (t.dtype() == torch::kBFloat16) {
    tensor_dtype = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  } else {
    TORCH_CHECK(false, "Invalid tensor dtype: ", t.dtype());
  }
  CUtensorMap tensor_map;
  cuuint64_t gmem_dims[] = {gmem_inner_dim, gmem_outer_dim};
  cuuint64_t gmem_strides[] = {gmem_outer_stride * t.element_size()};
  cuuint32_t smem_dims[] = {smem_inner_dim, smem_outer_dim};
  cuuint32_t smem_strides[] = {1, 1};
  // std::cout << "Make2DTMADesc, gmem_inner_dim: " << gmem_inner_dim
  //           << ", gmem_outer_dim: " << gmem_outer_dim
  //           << ", gmem_outer_stride: " << gmem_outer_stride * t.element_size()
  //           << ", smem_inner_dim : " << smem_inner_dim
  //           << ", smem_outer_dim: "
  //           << smem_outer_dim
  //           << ", swizzle_type: "
  //           << swizzle_type
  //           << std::endl;
  CUresult result =
      cuTensorMapEncodeTiled(&tensor_map, tensor_dtype, 2, t.data_ptr(), gmem_dims, gmem_strides,
                             smem_dims, smem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle_type,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", GetCUDAErrorName(result));
  return tensor_map;
}

CUtensorMapSwizzle GetSwizzleType(int swizzle_mode) {
  if (swizzle_mode == 0) {
    return CU_TENSOR_MAP_SWIZZLE_NONE;
  } else if (swizzle_mode == 32) {
    return CU_TENSOR_MAP_SWIZZLE_32B;
  } else if (swizzle_mode == 64) {
    return CU_TENSOR_MAP_SWIZZLE_64B;
  } else if (swizzle_mode == 128) {
    return CU_TENSOR_MAP_SWIZZLE_128B;
  } else {
    TORCH_CHECK(false, "Invalid swizzle mode: ", swizzle_mode);
  }
}

std::uintptr_t LoadKernel(std::string lib_path, std::string kernel_name) {
  CUlibrary lib;
  CUresult result =
      cuLibraryLoadFromFile(&lib, lib_path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0);
  TORCH_CHECK(result == CUDA_SUCCESS, "cuLibraryLoadFromFile failed: ", GetCUDAErrorName(result));
  CUkernel kernel;
  result = cuLibraryGetKernel(&kernel, lib, kernel_name.c_str());
  TORCH_CHECK(result == CUDA_SUCCESS, "cuLibraryGetKernel failed: ", GetCUDAErrorName(result));
  return reinterpret_cast<std::uintptr_t>(kernel);
}

void LaunchFP8GeMM(std::uintptr_t kernel_handle, torch::Tensor a, torch::Tensor b, torch::Tensor d,
                   torch::Tensor scales_a, torch::Tensor scales_b, torch::Tensor grouped_layout,
                   uint32_t m, uint32_t n, uint32_t k, uint32_t stride_am, uint32_t stride_bn,
                   uint32_t stride_dm, int block_m, int block_n, int block_k, int num_groups,
                   int d_swizzle_mode, int num_sms, int num_tma_multicast, int smem_size,
                   CUdevice device, std::uintptr_t stream_handle) {
  CUcontext ctx;
  CUresult result = cuCtxGetCurrent(&ctx);
  TORCH_CHECK(result == CUDA_SUCCESS, "cuCtxGetCurrent failed: ", GetCUDAErrorName(result));
  result = cuCtxPushCurrent(ctx);
  TORCH_CHECK(result == CUDA_SUCCESS, "cuCtxPushCurrent failed: ", GetCUDAErrorName(result));
  CUkernel kernel = reinterpret_cast<CUkernel>(kernel_handle);
  CUstream stream = reinterpret_cast<CUstream>(stream_handle);

  // std::cout << "LaunchFP8GeMM, kernel_handle: 0x" << std::hex << kernel_handle
  //           << ", kernel: " << kernel << ", ctx: " << ctx << ", ctx ptr: " << std::hex << ctx
  //           << std::dec << ", device: " << device
  //           << std::endl;
  GemmType gemm_type = kNormal;
  CUtensorMap tensor_map_a =
      MakeSingle2DTMADesc(a, k, m * (gemm_type == kGroupedMasked ? num_groups : 1), stride_am,
                          block_k, block_m, CU_TENSOR_MAP_SWIZZLE_128B);
  CUtensorMap tensor_map_b =
      MakeSingle2DTMADesc(b, k, n * (gemm_type != kNormal ? num_groups : 1), stride_bn, block_k,
                          block_n, CU_TENSOR_MAP_SWIZZLE_128B);
  CUtensorMap tensor_map_d =
      MakeSingle2DTMADesc(d, n, m * (gemm_type == kGroupedMasked ? num_groups : 1), stride_dm,
                          (d_swizzle_mode == 0 ? block_n : d_swizzle_mode / d.element_size()),
                          block_m, GetSwizzleType(d_swizzle_mode));
  uint32_t aligned_m = GetTMAAlignedSize(m, scales_a.element_size());
  CUtensorMap tensor_map_scales_a = MakeSingle2DTMADesc(
      scales_a, aligned_m,
      (k + block_k - 1) / block_k * (gemm_type == kGroupedMasked ? num_groups : 1), aligned_m,
      block_m, 1, CU_TENSOR_MAP_SWIZZLE_NONE);

  int num_tma_threads = 128;
  int num_math_threads_per_group = 128;

  result = cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size, kernel,
                                device);
  TORCH_CHECK(result == CUDA_SUCCESS,
              "Failed to set max dynamic shared memory size: ", GetCUDAErrorName(result));

  CUlaunchAttributeValue attr_val;
  attr_val.clusterDim.x = num_tma_multicast;
  attr_val.clusterDim.y = 1;
  attr_val.clusterDim.z = 1;
  CUlaunchAttribute attr;
  attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  attr.value = attr_val;

  CUlaunchConfig config;
  config.numAttrs = 1;
  config.attrs = &attr;
  config.gridDimX = num_sms;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = GetNumThreadsPerSM(num_tma_threads, num_math_threads_per_group, block_m);
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = smem_size;
  config.hStream = stream;

  float* scales_b_ptr = static_cast<float*>(scales_b.data_ptr());
  int32_t* grouped_layout_ptr = static_cast<int32_t*>(grouped_layout.data_ptr());
  void* kernel_params[] = {&scales_b_ptr, &grouped_layout_ptr,  &m,           &tensor_map_a,
                           &tensor_map_b, &tensor_map_scales_a, &tensor_map_d};
  result = cuLaunchKernelEx(&config, reinterpret_cast<CUfunction>(kernel), kernel_params, nullptr);
  TORCH_CHECK(result == CUDA_SUCCESS, "Failed to launch kernel: " + GetCUDAErrorName(result));

  CUcontext popped_ctx;
  cuCtxPopCurrent(&popped_ctx);
  TORCH_CHECK(ctx == popped_ctx, "Popped context does not match pushed context!");
}

void LaunchWGradFP8GeMM(std::uintptr_t kernel_handle, torch::Tensor a, torch::Tensor b,
                        torch::Tensor d, torch::Tensor scales_a, torch::Tensor scales_b, uint32_t m,
                        uint32_t n, uint32_t k, uint32_t stride_am, uint32_t stride_bn,
                        uint32_t stride_dm, int block_m, int block_n, int block_k, int num_groups,
                        int d_swizzle_mode, int num_sms, int num_tma_multicast, int smem_size,
                        CUdevice device, std::uintptr_t stream_handle) {
  CUcontext ctx;
  CUresult result = cuCtxGetCurrent(&ctx);
  TORCH_CHECK(result == CUDA_SUCCESS, "cuCtxGetCurrent failed: ", GetCUDAErrorName(result));
  result = cuCtxPushCurrent(ctx);
  TORCH_CHECK(result == CUDA_SUCCESS, "cuCtxPushCurrent failed: ", GetCUDAErrorName(result));
  CUkernel kernel = reinterpret_cast<CUkernel>(kernel_handle);
  CUstream stream = reinterpret_cast<CUstream>(stream_handle);

  // std::cout << "LaunchWGradFP8GeMM, kernel_handle: 0x" << std::hex << kernel_handle
  //           << ", kernel: " << kernel << ", ctx: " << ctx << ", ctx ptr: " << std::hex << ctx
  //           << std::dec << ", device: " << device
  //           << std::endl;
  GemmType gemm_type = kNormal;
  CUtensorMap tensor_map_a =
      MakeSingle2DTMADesc(a, k, m * (gemm_type == kGroupedMasked ? num_groups : 1), stride_am,
                          block_k, block_m, CU_TENSOR_MAP_SWIZZLE_128B);
  CUtensorMap tensor_map_b =
      MakeSingle2DTMADesc(b, k, n * (gemm_type != kNormal ? num_groups : 1), stride_bn, block_k,
                          block_n, CU_TENSOR_MAP_SWIZZLE_128B);
  CUtensorMap tensor_map_d =
      MakeSingle2DTMADesc(d, n, m * (gemm_type == kGroupedMasked ? num_groups : 1), stride_dm,
                          (d_swizzle_mode == 0 ? block_n : d_swizzle_mode / d.element_size()),
                          block_m, GetSwizzleType(d_swizzle_mode));
  uint32_t aligned_m = GetTMAAlignedSize(m, scales_a.element_size());
  uint32_t aligned_n = GetTMAAlignedSize(n, scales_b.element_size());
  CUtensorMap tensor_map_scales_a = MakeSingle2DTMADesc(
      scales_a, aligned_m,
      (k + block_k - 1) / block_k * (gemm_type == kGroupedMasked ? num_groups : 1), aligned_m,
      block_m, 1, CU_TENSOR_MAP_SWIZZLE_NONE);
  CUtensorMap tensor_map_scales_b = MakeSingle2DTMADesc(
      scales_b, aligned_n,
      (k + block_k - 1) / block_k * (gemm_type == kGroupedMasked ? num_groups : 1), aligned_n,
      block_n, 1, CU_TENSOR_MAP_SWIZZLE_NONE);

  int num_tma_threads = 128;
  int num_math_threads_per_group = 128;

  result = cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size, kernel,
                                device);
  TORCH_CHECK(result == CUDA_SUCCESS,
              "Failed to set max dynamic shared memory size: ", GetCUDAErrorName(result));

  CUlaunchAttributeValue attr_val;
  attr_val.clusterDim.x = num_tma_multicast;
  attr_val.clusterDim.y = 1;
  attr_val.clusterDim.z = 1;
  CUlaunchAttribute attr;
  attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  attr.value = attr_val;

  CUlaunchConfig config;
  config.numAttrs = 1;
  config.attrs = &attr;
  config.gridDimX = num_sms;
  config.gridDimY = 1;
  config.gridDimZ = 1;
  config.blockDimX = GetNumThreadsPerSM(num_tma_threads, num_math_threads_per_group, block_m);
  config.blockDimY = 1;
  config.blockDimZ = 1;
  config.sharedMemBytes = smem_size;
  config.hStream = stream;

  void* kernel_params[] = {
      &k, &tensor_map_a, &tensor_map_b, &tensor_map_scales_a, &tensor_map_scales_b, &tensor_map_d};
  result = cuLaunchKernelEx(&config, reinterpret_cast<CUfunction>(kernel), kernel_params, nullptr);
  TORCH_CHECK(result == CUDA_SUCCESS, "Failed to launch kernel: " + GetCUDAErrorName(result));

  CUcontext popped_ctx;
  cuCtxPopCurrent(&popped_ctx);
  TORCH_CHECK(ctx == popped_ctx, "Popped context does not match pushed context!");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("load_kernel", &LoadKernel, "Load kernel");
  m.def("launch_fp8_gemm", &LaunchFP8GeMM, "FP8 GEMM launcher");
  m.def("launch_wgrad_fp8_gemm", &LaunchWGradFP8GeMM, "WGrad FP8 GEMM launcher");
}

}  // namespace lotus_deepgemm
