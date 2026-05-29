// SPDX-License-Identifier: Apache-2.0
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "../dispatch_utils.h"
#include "quest_selection.h"

namespace vllm {

// One CUDA program per candidate. Each program:
//   - loads candidate_ids[blockIdx.x] = block_id
//   - loops over H_kv heads and G query groups
//   - for each (h, g, d) computes max(q*k_max, q*k_min) in fp32
//   - reduces inside the block over D using shared memory
//   - thread 0 writes one fp32 score per candidate
//
// Reads strides at runtime to tolerate non-contiguous inputs (matches
// the Triton kernel contract). H_kv, D, G are runtime ints — no
// template specialization in Phase D (deferred per plan scope).
template <typename scalar_t>
__global__ void quest_score_kernel(const scalar_t* __restrict__ query,
                                   const scalar_t* __restrict__ block_summary,
                                   const int32_t* __restrict__ candidate_ids,
                                   float* __restrict__ scores, const int H_kv,
                                   const int G, const int D,
                                   const int q_stride_h, const int q_stride_d,
                                   const int s_stride_b, const int s_stride_2,
                                   const int s_stride_h, const int s_stride_d) {
  const int pid = blockIdx.x;
  const int tid = threadIdx.x;
  const int block_id = candidate_ids[pid];

  // Per-thread accumulator. Each thread covers strided D positions.
  float acc = 0.0f;

  for (int h = 0; h < H_kv; ++h) {
    const int max_base =
        block_id * s_stride_b + 0 * s_stride_2 + h * s_stride_h;
    const int min_base =
        block_id * s_stride_b + 1 * s_stride_2 + h * s_stride_h;

    for (int g = 0; g < G; ++g) {
      const int q_base = (h * G + g) * q_stride_h;

      for (int d = tid; d < D; d += blockDim.x) {
        const float k_max =
            static_cast<float>(block_summary[max_base + d * s_stride_d]);
        const float k_min =
            static_cast<float>(block_summary[min_base + d * s_stride_d]);
        const float q = static_cast<float>(query[q_base + d * q_stride_d]);
        acc += fmaxf(q * k_max, q * k_min);
      }
    }
  }

  // Block-wide reduction over threads using shared memory.
  __shared__ float sbuf[32];
  // Warp reduce.
  unsigned mask = 0xffffffffu;
  for (int offset = 16; offset > 0; offset >>= 1) {
    acc += __shfl_down_sync(mask, acc, offset);
  }
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (lane == 0) sbuf[warp] = acc;
  __syncthreads();

  // First warp reduces the per-warp partials.
  if (warp == 0) {
    const int num_warps = (blockDim.x + 31) >> 5;
    float v = (tid < num_warps) ? sbuf[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      v += __shfl_down_sync(mask, v, offset);
    }
    if (tid == 0) scores[pid] = v;
  }
}

}  // namespace vllm

void quest_score(const torch::Tensor& query, const torch::Tensor& block_summary,
                 const torch::Tensor& candidate_ids, torch::Tensor& scores,
                 int64_t num_kv_groups) {
  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(block_summary.is_cuda(), "block_summary must be a CUDA tensor");
  TORCH_CHECK(candidate_ids.is_cuda(), "candidate_ids must be a CUDA tensor");
  TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");
  TORCH_CHECK(query.dim() == 2,
              "query must be 2-D [H_kv * G, D], got dim=", query.dim());
  TORCH_CHECK(block_summary.dim() == 4,
              "block_summary must be 4-D [B_total, 2, H_kv, D], got dim=",
              block_summary.dim());
  TORCH_CHECK(candidate_ids.dim() == 1, "candidate_ids must be 1-D");
  TORCH_CHECK(scores.dim() == 1, "scores must be 1-D");
  TORCH_CHECK(scores.scalar_type() == at::ScalarType::Float,
              "scores must be fp32");
  TORCH_CHECK(candidate_ids.scalar_type() == at::ScalarType::Int,
              "candidate_ids must be int32");
  TORCH_CHECK(block_summary.size(1) == 2,
              "block_summary.shape[1] must be 2 (amax/amin), got ",
              block_summary.size(1));
  TORCH_CHECK(query.scalar_type() == block_summary.scalar_type(),
              "query and block_summary must share dtype");

  const int H_kv = static_cast<int>(block_summary.size(2));
  const int D = static_cast<int>(block_summary.size(3));
  const int G = static_cast<int>(num_kv_groups);
  const int B = static_cast<int>(candidate_ids.size(0));

  TORCH_CHECK(query.size(0) == H_kv * G, "query.shape[0] (", query.size(0),
              ") must equal H_kv*G (", H_kv * G, ")");
  TORCH_CHECK(query.size(1) == D, "query.shape[1] (", query.size(1),
              ") must equal head_size (", D, ")");
  TORCH_CHECK(scores.size(0) == B, "scores.shape[0] (", scores.size(0),
              ") must equal num_candidates (", B, ")");

  if (B == 0) return;

  const c10::cuda::OptionalCUDAGuard device_guard(query.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  constexpr int NUM_THREADS = 128;
  dim3 grid(B);
  dim3 block(NUM_THREADS);

  VLLM_DISPATCH_FLOATING_TYPES(query.scalar_type(), "quest_score", [&] {
    vllm::quest_score_kernel<scalar_t><<<grid, block, 0, stream>>>(
        query.data_ptr<scalar_t>(), block_summary.data_ptr<scalar_t>(),
        candidate_ids.data_ptr<int32_t>(), scores.data_ptr<float>(), H_kv, G, D,
        static_cast<int>(query.stride(0)), static_cast<int>(query.stride(1)),
        static_cast<int>(block_summary.stride(0)),
        static_cast<int>(block_summary.stride(1)),
        static_cast<int>(block_summary.stride(2)),
        static_cast<int>(block_summary.stride(3)));
  });
}
