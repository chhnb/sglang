#include <sgl_kernel/tensor.h>   // TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.cuh>  // LaunchKernel
#include <sgl_kernel/utils.h>    // RuntimeCheck
#include <sgl_kernel/vec.cuh>    // aligned_vector

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace {

enum NormType : int {
  LayerNorm = 0,
  RMSNorm = 1,
};

template <typename T>
using Vec4 = device::aligned_vector<T, 4>;

template <typename T, int NumVals>
__device__ __forceinline__ void warpReduceSum(T (&vals)[NumVals]) {
  unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int i = 0; i < NumVals; ++i) {
      vals[i] += __shfl_down_sync(mask, vals[i], offset);
    }
  }
}

struct WelfordData {
  float mean;
  float m2;
  float count;
};

__device__ __forceinline__ WelfordData welford_combine(WelfordData a, WelfordData b) {
  if (b.count == 0.0f) return a;
  if (a.count == 0.0f) return b;
  float delta = b.mean - a.mean;
  float count = a.count + b.count;
  float mean = a.mean + delta * (b.count / count);
  float m2 = a.m2 + b.m2 + delta * delta * (a.count * b.count / count);
  return {mean, m2, count};
}

__device__ __forceinline__ void welford_update(WelfordData& w, float x) {
  float count = w.count + 1.0f;
  float delta = x - w.mean;
  float mean = w.mean + delta / count;
  float delta2 = x - mean;
  w.mean = mean;
  w.m2 += delta * delta2;
  w.count = count;
}

__device__ __forceinline__ WelfordData warpReduceWelford(WelfordData w) {
  unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    WelfordData other;
    other.mean = __shfl_down_sync(mask, w.mean, offset);
    other.m2 = __shfl_down_sync(mask, w.m2, offset);
    other.count = __shfl_down_sync(mask, w.count, offset);
    w = welford_combine(w, other);
  }
  return w;
}

__device__ __forceinline__ WelfordData blockReduceWelford(WelfordData w) {
  __shared__ WelfordData shared[32];
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  w = warpReduceWelford(w);
  if (lane == 0) {
    shared[wid] = w;
  }
  __syncthreads();
  if (wid == 0) {
    int num_warps = (blockDim.x + 31) / 32;
    WelfordData acc;
    acc.mean = 0.0f;
    acc.m2 = 0.0f;
    acc.count = 0.0f;
    if (lane < num_warps) {
      acc = shared[lane];
    }
    acc = warpReduceWelford(acc);
    if (lane == 0) {
      shared[0] = acc;
    }
  }
  __syncthreads();
  return shared[0];
}

template <typename T, int NumVals>
__device__ __forceinline__ void blockReduceSum(T (&vals)[NumVals]) {
  __shared__ T shared[32][NumVals];  // up to 32 warps (1024 threads)
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  warpReduceSum<T, NumVals>(vals);
  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NumVals; ++i) {
      shared[wid][i] = vals[i];
    }
  }
  __syncthreads();
  if (wid == 0) {
    int num_warps = (blockDim.x + 31) / 32;
    T acc[NumVals];
#pragma unroll
    for (int i = 0; i < NumVals; ++i) {
      acc[i] = (lane < num_warps) ? shared[lane][i] : T(0);
    }
    warpReduceSum<T, NumVals>(acc);
#pragma unroll
    if (lane == 0) {
#pragma unroll
      for (int i = 0; i < NumVals; ++i) {
        shared[0][i] = acc[i];
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < NumVals; ++i) {
    vals[i] = shared[0][i];
  }
}

template <typename T>
__device__ __forceinline__ float vec4_sum(const Vec4<T>& v) {
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    sum += static_cast<float>(v[j]);
  }
  return sum;
}

template <typename T>
__device__ __forceinline__ float vec4_sum_sq(const Vec4<T>& v) {
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float val = static_cast<float>(v[j]);
    sum += val * val;
  }
  return sum;
}

template <typename T>
__device__ __forceinline__ float vec4_variance_sum(const Vec4<T>& v, float mean) {
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float diff = static_cast<float>(v[j]) - mean;
    sum += diff * diff;
  }
  return sum;
}

// kScaleBroadcast == true  -> scale is [1, N]
// kScaleBroadcast == false -> scale is [B, N]
template <int kNormType, typename T, bool kScaleBroadcast, int ITEM_PER_THREAD>
__global__ void norm_scale_fused_kernel(
    T* __restrict__ out,
    const T* __restrict__ x,
    const T* __restrict__ scale,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    int64_t M,
    int64_t N,
    int64_t L,
    float add_const,
    float eps) {
  int64_t row = blockIdx.x;
  if (row >= M) return;

  int tid = threadIdx.x;
  int64_t n4 = N / 4;
  int64_t row_off = row * n4;

  int64_t b = row / L;
  int64_t scale_row = kScaleBroadcast ? 0 : b;
  const T* scale_row_ptr = scale + scale_row * N;

  __shared__ float s_mean;
  __shared__ float s_rstd;

  float local_sum[1] = {0.0f};
  WelfordData local_w;
  local_w.mean = 0.0f;
  local_w.m2 = 0.0f;
  local_w.count = 0.0f;
  Vec4<T> local_val[ITEM_PER_THREAD];

#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; ++i) {
    int64_t idx = static_cast<int64_t>(i) * blockDim.x + tid;
    if (idx < n4) {
      local_val[i].load(x, row_off + idx);
    } else {
      local_val[i].fill(T(0));
    }
    if constexpr (kNormType == LayerNorm) {
      if (idx < n4) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          welford_update(local_w, static_cast<float>(local_val[i][j]));
        }
      }
    } else {
      local_sum[0] += vec4_sum_sq(local_val[i]);
    }
  }

  if constexpr (kNormType == LayerNorm) {
    WelfordData total = blockReduceWelford(local_w);
    if (tid == 0) {
      s_mean = total.mean;
      float denom = total.m2 / static_cast<float>(N);
      s_rstd = rsqrtf(denom + eps);
    }
    __syncthreads();
  } else {
    if (blockDim.x <= 32) {
      warpReduceSum<float, 1>(local_sum);
    } else {
      blockReduceSum<float, 1>(local_sum);
    }
    if (tid == 0) {
      float denom = local_sum[0] / static_cast<float>(N);
      s_rstd = rsqrtf(denom + eps);
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < ITEM_PER_THREAD; ++i) {
    int64_t idx = static_cast<int64_t>(i) * blockDim.x + tid;
    if (idx < n4) {
      Vec4<T> out_v;
      Vec4<T> w_v;
      Vec4<T> sc_v;
      Vec4<T> b_v;
      w_v.load(weight, idx);
      sc_v.load(scale_row_ptr, idx);
      if constexpr (kNormType == LayerNorm) {
        b_v.load(bias, idx);
      } else {
        b_v.fill(T(0));
      }

#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float x_f = static_cast<float>(local_val[i][j]);
        float norm = (kNormType == LayerNorm) ? ((x_f - s_mean) * s_rstd) : (x_f * s_rstd);
        float affine = norm * static_cast<float>(w_v[j]) + static_cast<float>(b_v[j]);
        float sc = static_cast<float>(sc_v[j]) + add_const;
        out_v[j] = static_cast<T>(affine * sc);
      }
      out_v.store(out, row_off + idx);
    }
  }
}

template <int kNormType, bool kScaleBroadcast, int ITEM_PER_THREAD>
void norm_scale_fused(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView scale,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView bias,
    int64_t L,
    double add_const,
    double eps) {
  using namespace host;

  SymbolicSize M_ = {"M"};
  SymbolicSize N_ = {"N"};
  SymbolicDevice device_;
  TensorMatcher({M_, N_})
      .with_dtype<float, half, nv_bfloat16>()
      .with_strides({N_, 1})
      .with_device<kDLCUDA>(device_)
      .verify(out)
      .verify(x);

  TensorMatcher({details::kAnySize, N_})
      .with_dtype<float, half, nv_bfloat16>()
      .with_strides({N_, 1})
      .with_device<kDLCUDA>(device_)
      .verify(scale);

  TensorMatcher({N_}).with_dtype<float, half, nv_bfloat16>().with_device<kDLCUDA>(device_).verify(weight);

  TensorMatcher({N_}).with_dtype<float, half, nv_bfloat16>().with_device<kDLCUDA>(device_).verify(bias);

  const int64_t M = M_.unwrap();
  const int64_t N = N_.unwrap();
  const auto dtype = x.dtype();
  auto same_dtype = [&](const DLDataType& other) {
    return dtype.code == other.code && dtype.bits == other.bits && dtype.lanes == other.lanes;
  };
  RuntimeCheck(
      same_dtype(scale.dtype()) && same_dtype(weight.dtype()) && same_dtype(bias.dtype()),
      "x/scale/weight/bias must have the same dtype");
  RuntimeCheck(N % 4 == 0, "N must be divisible by 4.");
  RuntimeCheck(L > 0, "L must be > 0.");
  RuntimeCheck(M % L == 0, "M must be divisible by L.");

  const int64_t B = M / L;
  const int64_t scale_B = scale.size(0);
  RuntimeCheck(scale_B == 1 || scale_B == B, "scale must have B==1 or B==M/L.");

  int64_t n4 = N / 4;
  int threads = static_cast<int>((n4 + ITEM_PER_THREAD - 1) / ITEM_PER_THREAD);
  threads = (threads + 31) / 32 * 32;
  if (threads > 1024) threads = 1024;

  dim3 grid(static_cast<unsigned>(M));
  dim3 block(threads);

  auto launch = [&]<typename T>() {
    LaunchKernel(grid, block, out.device())(
        norm_scale_fused_kernel<kNormType, T, kScaleBroadcast, ITEM_PER_THREAD>,
        static_cast<T*>(out.data_ptr()),
        static_cast<const T*>(x.data_ptr()),
        static_cast<const T*>(scale.data_ptr()),
        static_cast<const T*>(weight.data_ptr()),
        static_cast<const T*>(bias.data_ptr()),
        M,
        N,
        L,
        static_cast<float>(add_const),
        static_cast<float>(eps));
  };

  if (dtype.code == kDLFloat && dtype.bits == 32) {
    launch.template operator()<float>();
  } else if (dtype.code == kDLBfloat && dtype.bits == 16) {
    launch.template operator()<__nv_bfloat16>();
  } else if (dtype.code == kDLFloat && dtype.bits == 16) {
    launch.template operator()<__half>();
  } else {
    RuntimeCheck(false, "Unsupported dtype.");
  }
}

}  // namespace
