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
#include "mobilenet_param_set.h"
#include "snn_fixture.h"

#include "src/backend/snn_backend_provider.h"

#include "sycldnn/backend/snn_backend.h"

#include "sycldnn/pooling/operators.h"

#define MOBILENET_BM_WITH_DIR_AND_OP(N, C, W, H, K, S, DIRECTION, OP)      \
  POOLING_BENCHMARK(                                                       \
      "MobileNet",                                                         \
      OP##_##DIRECTION##_##N##_##C##_##W##_##H##_##K##_##S##_##SNNBackend, \
      sycldnn::backend::SNNBackend, ParameterSet<N, C, W, H, K, S>,        \
      sycldnn::pooling::DIRECTION, sycldnn::pooling::OP)

#define MOBILENET_BM_WITH_DIRECTION(N, C, W, H, K, S, DIRECTION) \
  MOBILENET_BM_WITH_DIR_AND_OP(N, C, W, H, K, S, DIRECTION, Max) \
  MOBILENET_BM_WITH_DIR_AND_OP(N, C, W, H, K, S, DIRECTION, Average)

#define MOBILENET_BENCHMARK(N, C, W, H, K, S)            \
  MOBILENET_BM_WITH_DIRECTION(N, C, W, H, K, S, Forward) \
  MOBILENET_BM_WITH_DIRECTION(N, C, W, H, K, S, Backpropagate)

// Standard benchmark sizes (batch size: 1, 4, optionally 32
#define MOBILENET_PARAMS(channels, width, height, window, stride) \
  MOBILENET_BENCHMARK(1, channels, width, height, window, stride);
#include "bench/pooling/mobilenet_params.def"
#undef MOBILENET_PARAMS

#define MOBILENET_PARAMS(channels, width, height, window, stride) \
  MOBILENET_BENCHMARK(4, channels, width, height, window, stride);
#include "bench/pooling/mobilenet_params.def"
#undef MOBILENET_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define MOBILENET_PARAMS(channels, width, height, window, stride) \
  MOBILENET_BENCHMARK(32, channels, width, height, window, stride);
#include "bench/pooling/mobilenet_params.def"
#undef MOBILENET_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS

// Extended benchmarks (batch size: 2, optionally 8, 16, 64)
#ifdef SNN_EXTENDED_BENCHMARKS
#define MOBILENET_PARAMS(channels, width, height, window, stride) \
  MOBILENET_BENCHMARK(2, channels, width, height, window, stride);
#include "bench/pooling/mobilenet_params.def"
#undef MOBILENET_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define MOBILENET_PARAMS(channels, width, height, window, stride) \
  MOBILENET_BENCHMARK(8, channels, width, height, window, stride);
#include "bench/pooling/mobilenet_params.def"
#undef MOBILENET_PARAMS

#define MOBILENET_PARAMS(channels, width, height, window, stride) \
  MOBILENET_BENCHMARK(16, channels, width, height, window, stride);
#include "bench/pooling/mobilenet_params.def"
#undef MOBILENET_PARAMS

#define MOBILENET_PARAMS(channels, width, height, window, stride) \
  MOBILENET_BENCHMARK(64, channels, width, height, window, stride);
#include "bench/pooling/mobilenet_params.def"
#undef MOBILENET_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS
#endif  // SNN_EXTENDED_BENCHMARKS
