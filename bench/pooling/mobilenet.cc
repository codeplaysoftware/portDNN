/*
 * Copyright 2018 Codeplay Software Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use these files except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "param_set.h"
#include "snn_fixture.h"

#include "src/backend/snn_backend_provider.h"

#include "sycldnn/backend/snn_backend.h"

#include "sycldnn/pooling/operators.h"

#define MOBILENET_BM_WITH_DIR_OP_DTYPE(N, C, H, W, K, S, PAD, DIR, OP, DTYPE) \
  POOLING_BENCHMARK(                                                          \
      "MobileNet",                                                            \
      OP##_##DIR##_##N##_##C##_##H##_##W##_##K##_##S##_##SNNBackend,          \
      sycldnn::backend::SNNBackend, DTYPE,                                    \
      ParameterSet<N, C, H, W, K, S, PAD>, sycldnn::pooling::DIR,             \
      sycldnn::pooling::OP)

#define MOBILENET_BM_WITH_DIR_OP(N, C, H, W, K, S, PAD, DIR, OP) \
  MOBILENET_BM_WITH_DIR_OP_DTYPE(N, C, H, W, K, S, PAD, DIR, OP, float)

#define MOBILENET_BM_WITH_DIRECTION(N, C, H, W, K, S, PAD, DIR) \
  MOBILENET_BM_WITH_DIR_OP(N, C, H, W, K, S, PAD, DIR, Max)     \
  MOBILENET_BM_WITH_DIR_OP(N, C, H, W, K, S, PAD, DIR, Average)

#define MOBILENET_BENCHMARK(N, C, H, W, K, S, PAD)            \
  MOBILENET_BM_WITH_DIRECTION(N, C, H, W, K, S, PAD, Forward) \
  MOBILENET_BM_WITH_DIRECTION(N, C, H, W, K, S, PAD, Backpropagate)

// Standard benchmark sizes (batch size: 1, 4, optionally 32
#define MOBILENET_PARAMS(C, H, W, K, S, PAD) \
  MOBILENET_BENCHMARK(1, C, H, W, K, S, PAD);
#include "bench/pooling/mobilenet_params.def"
#undef MOBILENET_PARAMS

#define MOBILENET_PARAMS(C, H, W, K, S, PAD) \
  MOBILENET_BENCHMARK(4, C, H, W, K, S, PAD);
#include "bench/pooling/mobilenet_params.def"
#undef MOBILENET_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define MOBILENET_PARAMS(C, H, W, K, S, PAD) \
  MOBILENET_BENCHMARK(32, C, H, W, K, S, PAD);
#include "bench/pooling/mobilenet_params.def"
#undef MOBILENET_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS

// Extended benchmarks (batch size: 2, optionally 8, 16, 64)
#ifdef SNN_EXTENDED_BENCHMARKS
#define MOBILENET_PARAMS(C, H, W, K, S, PAD) \
  MOBILENET_BENCHMARK(2, C, H, W, K, S, PAD);
#include "bench/pooling/mobilenet_params.def"
#undef MOBILENET_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define MOBILENET_PARAMS(C, H, W, K, S, PAD) \
  MOBILENET_BENCHMARK(8, C, H, W, K, S, PAD);
#include "bench/pooling/mobilenet_params.def"
#undef MOBILENET_PARAMS

#define MOBILENET_PARAMS(C, H, W, K, S, PAD) \
  MOBILENET_BENCHMARK(16, C, H, W, K, S, PAD);
#include "bench/pooling/mobilenet_params.def"
#undef MOBILENET_PARAMS

#define MOBILENET_PARAMS(C, H, W, K, S, PAD) \
  MOBILENET_BENCHMARK(64, C, H, W, K, S, PAD);
#include "bench/pooling/mobilenet_params.def"
#undef MOBILENET_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS
#endif  // SNN_EXTENDED_BENCHMARKS
