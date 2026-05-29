// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <torch/all.h>

// Quest block-selection score reduction.
//
// Computes per-candidate scalar scores:
//
//   score(b) = sum_{h, g, d} max(q[h*G+g, d] * k_max[b, h, d],
//                                q[h*G+g, d] * k_min[b, h, d])
//
// where b ranges over candidate_ids, k_max = block_summary[*, 0],
// k_min = block_summary[*, 1]. Reductions are always fp32; inputs may
// be fp16 or bf16. Top-k itself runs from Python (torch.topk on the
// returned scores tensor).
//
// Shapes:
//   query           [H_kv * G, D]                      (fp16/bf16)
//   block_summary   [B_total, 2, H_kv, D]              (fp16/bf16)
//   candidate_ids   [B]                                (int32)
//   scores          [B]                                (fp32, output)
void quest_score(const torch::Tensor& query, const torch::Tensor& block_summary,
                 const torch::Tensor& candidate_ids, torch::Tensor& scores,
                 int64_t num_kv_groups);
