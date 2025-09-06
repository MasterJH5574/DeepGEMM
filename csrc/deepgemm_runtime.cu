#include <ATen/Operators.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <torch/all.h>
#include <torch/extension.h>
#include <torch/library.h>

namespace deepgemm_runtime {

inline int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

inline std::string GetCUDAErrorName(CUresult result) {
  const char* error_name = nullptr;
  cuGetErrorName(result, &error_name);
  return std::string(error_name);
}

inline int64_t GetTMAAlignedSize(int64_t x, int64_t element_size) {
  int64_t tma_alignment_bytes = 16;
  TORCH_CHECK(tma_alignment_bytes % element_size == 0,
              "TMA alignment bytes must be divisible by element size");
  int64_t alignment = tma_alignment_bytes / element_size;
  return (x + alignment - 1) / alignment * alignment;
}

inline bool IsColMajorTMAAligned(at::Tensor x) {
  if (x.dim() == 2) {
    int64_t m = x.size(0);
    int64_t aligned_m = GetTMAAlignedSize(m, x.element_size());
    return x.stride(0) == 1 && x.stride(1) == aligned_m;
  } else if (x.dim() == 3) {
    int64_t b = x.size(0);
    int64_t m = x.size(1);
    int64_t n = x.size(2);
    int64_t aligned_m = GetTMAAlignedSize(m, x.element_size());
    return x.stride(0) == aligned_m * n && x.stride(1) == 1 && x.stride(2) == aligned_m;
  } else {
    TORCH_CHECK(false, "Invalid tensor dimension: ", x.dim());
  }
}

inline int GetNumMathWarpGroups(int64_t block_m) { return block_m == 64 ? 1 : 2; }

inline int GetNumThreadsPerSM(int num_tma_threads, int num_math_threads_per_group,
                              int64_t block_m) {
  TORCH_CHECK(num_math_threads_per_group == 128, "Only support 128 threads per math group");
  return GetNumMathWarpGroups(block_m) * num_math_threads_per_group + num_tma_threads;
}

enum GemmType : int {
  kNormal = 0,
  kGroupedContiguous = 1,
  kGroupedMasked = 2,
};

CUtensorMap MakeSingle2DTMADesc(at::Tensor t, cuuint64_t gmem_inner_dim, cuuint64_t gmem_outer_dim,
                                cuuint64_t gmem_outer_stride, cuuint32_t smem_inner_dim,
                                cuuint32_t smem_outer_dim, CUtensorMapSwizzle swizzle_type) {
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

int64_t LoadKernel(std::string lib_path, std::string kernel_name) {
  CUlibrary lib;
  CUresult result =
      cuLibraryLoadFromFile(&lib, lib_path.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0);
  TORCH_CHECK(result == CUDA_SUCCESS, "cuLibraryLoadFromFile failed: ", GetCUDAErrorName(result));
  CUkernel kernel;
  result = cuLibraryGetKernel(&kernel, lib, kernel_name.c_str());
  TORCH_CHECK(result == CUDA_SUCCESS, "cuLibraryGetKernel failed: ", GetCUDAErrorName(result));
  return reinterpret_cast<int64_t>(kernel);
}

at::Tensor LaunchFP8GeMM(int64_t kernel_handle, at::Tensor a, at::Tensor b, at::Tensor scales_a,
                         at::Tensor scales_b, int64_t block_m, int64_t block_n, int64_t block_k,
                         int64_t num_groups, int64_t d_swizzle_mode, int64_t num_sms,
                         int64_t num_tma_multicast, int64_t smem_size) {
  auto device = a.device();
  auto a_sizes = a.sizes();
  int64_t m = a.size(0);
  int64_t k = a.size(1);
  int64_t n = b.size(0);
  int64_t k_ = b.size(1);
  at::Tensor d = at::empty({m, n}, at::TensorOptions().dtype(at::kBFloat16).device(device));
  at::Tensor grouped_layout = at::empty({0}, at::TensorOptions().dtype(at::kInt).device(device));
  int64_t stride_am = a.stride(0);
  int64_t stride_bn = b.stride(0);
  int64_t stride_dm = d.stride(0);

  if (m == 0) {
    return d;
  }

  TORCH_CHECK(k == k_);
  TORCH_CHECK(n > 0 && k > 0);
  TORCH_CHECK(num_groups == 1);
  TORCH_CHECK(scales_a.sizes() == c10::IntArrayRef({m, ceil_div(k, 128)}));
  TORCH_CHECK(scales_b.sizes() == c10::IntArrayRef({ceil_div(n, 128), ceil_div(k, 128)}));
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(scales_a.scalar_type() == torch::kFloat32);
  TORCH_CHECK(scales_b.scalar_type() == torch::kFloat32);
  TORCH_CHECK(a.stride(1) == 1);
  TORCH_CHECK(b.stride(1) == 1);
  TORCH_CHECK(IsColMajorTMAAligned(scales_a),
              "scales_a is not col major tma aligned, with strides (", scales_a.stride(0), ", ",
              scales_a.stride(1), "), shape (", scales_a.size(0), ", ", scales_a.size(1), ")");
  TORCH_CHECK(scales_b.is_contiguous());

  CUcontext ctx;
  CUresult result = cuCtxGetCurrent(&ctx);
  TORCH_CHECK(result == CUDA_SUCCESS, "cuCtxGetCurrent failed: ", GetCUDAErrorName(result));
  result = cuCtxPushCurrent(ctx);
  TORCH_CHECK(result == CUDA_SUCCESS, "cuCtxPushCurrent failed: ", GetCUDAErrorName(result));
  CUkernel kernel = reinterpret_cast<CUkernel>(kernel_handle);
  CUstream stream = at::cuda::getCurrentCUDAStream();

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
  int64_t aligned_m = GetTMAAlignedSize(m, scales_a.element_size());
  CUtensorMap tensor_map_scales_a = MakeSingle2DTMADesc(
      scales_a, aligned_m,
      (k + block_k - 1) / block_k * (gemm_type == kGroupedMasked ? num_groups : 1), aligned_m,
      block_m, 1, CU_TENSOR_MAP_SWIZZLE_NONE);

  int num_tma_threads = 128;
  int num_math_threads_per_group = 128;

  result = cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size, kernel,
                                static_cast<CUdevice>(device.index()));
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
  return d;
}

at::Tensor LaunchMGroupedFP8GeMM(int64_t kernel_handle, at::Tensor a, at::Tensor b,
                                 at::Tensor scales_a, at::Tensor scales_b,
                                 at::Tensor grouped_layout, int64_t block_m, int64_t block_n,
                                 int64_t block_k, int64_t num_groups, int64_t d_swizzle_mode,
                                 int64_t num_sms, int64_t num_tma_multicast, int64_t smem_size) {
  auto device = a.device();
  auto a_sizes = a.sizes();
  int64_t m = a.size(0);
  int64_t k = a.size(1);
  int64_t n = b.size(1);
  int64_t k_ = b.size(2);
  at::Tensor d = at::empty({m, n}, at::TensorOptions().dtype(at::kBFloat16).device(device));

  if (m == 0) {
    return d;
  }

  TORCH_CHECK(k == k_);
  TORCH_CHECK(n > 0 && k > 0);
  TORCH_CHECK(b.size(0) == num_groups);
  TORCH_CHECK(scales_a.sizes() == c10::IntArrayRef({m, ceil_div(k, 128)}));
  TORCH_CHECK(scales_b.sizes() ==
              c10::IntArrayRef({num_groups, ceil_div(n, 128), ceil_div(k, 128)}));
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(scales_a.scalar_type() == torch::kFloat32);
  TORCH_CHECK(scales_b.scalar_type() == torch::kFloat32);
  TORCH_CHECK(a.is_contiguous());
  TORCH_CHECK(b.is_contiguous());
  TORCH_CHECK(IsColMajorTMAAligned(scales_a),
              "scales_a is not col major tma aligned, with strides (", scales_a.stride(0), ", ",
              scales_a.stride(1), "), shape (", scales_a.size(0), ", ", scales_a.size(1), ")");
  TORCH_CHECK(scales_b.is_contiguous());
  TORCH_CHECK(grouped_layout.scalar_type() == at::kInt);
  TORCH_CHECK(grouped_layout.is_contiguous());
  TORCH_CHECK(grouped_layout.size(0) == m);

  CUcontext ctx;
  CUresult result = cuCtxGetCurrent(&ctx);
  TORCH_CHECK(result == CUDA_SUCCESS, "cuCtxGetCurrent failed: ", GetCUDAErrorName(result));
  result = cuCtxPushCurrent(ctx);
  TORCH_CHECK(result == CUDA_SUCCESS, "cuCtxPushCurrent failed: ", GetCUDAErrorName(result));
  CUkernel kernel = reinterpret_cast<CUkernel>(kernel_handle);
  CUstream stream = at::cuda::getCurrentCUDAStream();

  GemmType gemm_type = kGroupedContiguous;
  CUtensorMap tensor_map_a =
      MakeSingle2DTMADesc(a, k, m * (gemm_type == kGroupedMasked ? num_groups : 1), k, block_k,
                          block_m, CU_TENSOR_MAP_SWIZZLE_128B);
  CUtensorMap tensor_map_b = MakeSingle2DTMADesc(b, k, n * (gemm_type != kNormal ? num_groups : 1),
                                                 k, block_k, block_n, CU_TENSOR_MAP_SWIZZLE_128B);
  CUtensorMap tensor_map_d =
      MakeSingle2DTMADesc(d, n, m * (gemm_type == kGroupedMasked ? num_groups : 1), n,
                          (d_swizzle_mode == 0 ? block_n : d_swizzle_mode / d.element_size()),
                          block_m, GetSwizzleType(d_swizzle_mode));
  int64_t aligned_m = GetTMAAlignedSize(m, scales_a.element_size());
  CUtensorMap tensor_map_scales_a = MakeSingle2DTMADesc(
      scales_a, aligned_m,
      (k + block_k - 1) / block_k * (gemm_type == kGroupedMasked ? num_groups : 1), aligned_m,
      block_m, 1, CU_TENSOR_MAP_SWIZZLE_NONE);

  int num_tma_threads = 128;
  int num_math_threads_per_group = 128;

  result = cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size, kernel,
                                static_cast<CUdevice>(device.index()));
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
  return d;
}

void LaunchWGradFP8GeMM(int64_t kernel_handle, at::Tensor a, at::Tensor b, at::Tensor scales_a,
                        at::Tensor scales_b, at::Tensor d, int64_t block_m, int64_t block_n,
                        int64_t block_k, int64_t num_groups, int64_t d_swizzle_mode,
                        int64_t num_sms, int64_t num_tma_multicast, int64_t smem_size) {
  auto device = a.device();
  auto a_sizes = a.sizes();
  int64_t m = a.size(0);
  int64_t k = a.size(1);
  int64_t n = b.size(0);
  int64_t k_ = b.size(1);
  int64_t stride_am = a.stride(0);
  int64_t stride_bn = b.stride(0);
  int64_t stride_dm = d.stride(0);

  if (k == 0) {
    return;
  }

  TORCH_CHECK(k == k_);
  TORCH_CHECK(m > 0 && n > 0 && k > 0);
  TORCH_CHECK(scales_a.sizes() == c10::IntArrayRef({m, ceil_div(k, 128)}));
  TORCH_CHECK(scales_b.sizes() == c10::IntArrayRef({n, ceil_div(k, 128)}));
  TORCH_CHECK(a.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(b.scalar_type() == torch::kFloat8_e4m3fn);
  TORCH_CHECK(scales_a.scalar_type() == torch::kFloat32);
  TORCH_CHECK(scales_b.scalar_type() == torch::kFloat32);
  TORCH_CHECK(a.stride(1) == 1);
  TORCH_CHECK(b.stride(1) == 1);
  TORCH_CHECK(IsColMajorTMAAligned(scales_a),
              "scales_a is not col major tma aligned, with strides (", scales_a.stride(0), ", ",
              scales_a.stride(1), ")");
  TORCH_CHECK(IsColMajorTMAAligned(scales_b),
              "scales_b is not col major tma aligned, with strides (", scales_b.stride(0), ", ",
              scales_b.stride(1), ")");

  CUcontext ctx;
  CUresult result = cuCtxGetCurrent(&ctx);
  TORCH_CHECK(result == CUDA_SUCCESS, "cuCtxGetCurrent failed: ", GetCUDAErrorName(result));
  result = cuCtxPushCurrent(ctx);
  TORCH_CHECK(result == CUDA_SUCCESS, "cuCtxPushCurrent failed: ", GetCUDAErrorName(result));
  CUkernel kernel = reinterpret_cast<CUkernel>(kernel_handle);
  CUstream stream = at::cuda::getCurrentCUDAStream();

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
  int64_t aligned_m = GetTMAAlignedSize(m, scales_a.element_size());
  int64_t aligned_n = GetTMAAlignedSize(n, scales_b.element_size());
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
                                static_cast<CUdevice>(device.index()));
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

}  // namespace deepgemm_runtime

TORCH_LIBRARY(deepgemm_runtime, m) {
  m.def("load_kernel(str lib_path, str kernel_name) -> int");
  m.def(
      "launch_fp8_gemm(int kernel_handle, Tensor a, Tensor b, Tensor scales_a, Tensor scales_b, "
      "int block_m, int block_n, int block_k, "
      "int num_groups, int d_swizzle_mode, int num_sms, int num_tma_multicast, int smem_size) -> "
      "Tensor");
  m.def(
      "launch_m_grouped_fp8_gemm(int kernel_handle, Tensor a, Tensor b, Tensor scales_a, Tensor "
      "scales_b, Tensor grouped_layout, int block_m, int block_n, int block_k, "
      "int num_groups, int d_swizzle_mode, int num_sms, "
      "int num_tma_multicast, int smem_size) -> Tensor");
  m.def(
      "launch_wgrad_fp8_gemm(int kernel_handle, Tensor a, Tensor b, Tensor scales_a, "
      "Tensor scales_b, Tensor(d!) d, int block_m, int block_n, int block_k, "
      "int num_groups, int d_swizzle_mode, int num_sms, int num_tma_multicast, int smem_size) -> "
      "()");
}

TORCH_LIBRARY_IMPL(deepgemm_runtime, CatchAll, m) {
  m.impl("load_kernel", &deepgemm_runtime::LoadKernel);
}

TORCH_LIBRARY_IMPL(deepgemm_runtime, CUDA, m) {
  m.impl("launch_fp8_gemm", &deepgemm_runtime::LaunchFP8GeMM);
  m.impl("launch_m_grouped_fp8_gemm", &deepgemm_runtime::LaunchMGroupedFP8GeMM);
  m.impl("launch_wgrad_fp8_gemm", &deepgemm_runtime::LaunchWGradFP8GeMM);
}

PYBIND11_MODULE(deepgemm_runtime, m) {
  // Nothing else needed here, since ops are registered by the TORCH_LIBRARY macros
  // We need "PYBIND11_MODULE" to ensure pybind initialization.
}
