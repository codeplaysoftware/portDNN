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
#include "resnet_param_set.h"
#include "snn_fixture.h"

#include "src/backend/eigen_backend_provider.h"

#include "sycldnn/backend/eigen_backend.h"

#include "sycldnn/pooling/operators.h"

#define RESNET_BM_WITH_DIR_AND_OP(N, C, W, H, K, S, DIRECTION, OP)           \
  POOLING_BENCHMARK(                                                         \
      "ResNet",                                                              \
      OP##_##DIRECTION##_##N##_##C##_##W##_##H##_##K##_##S##_##EigenBackend, \
      sycldnn::backend::EigenBackend, ParameterSet<N, C, W, H, K, S>,        \
      sycldnn::pooling::DIRECTION, sycldnn::pooling::OP)

#define RESNET_BM_WITH_DIRECTION(N, C, W, H, K, S, DIRECTION) \
  RESNET_BM_WITH_DIR_AND_OP(N, C, W, H, K, S, DIRECTION, Max) \
  RESNET_BM_WITH_DIR_AND_OP(N, C, W, H, K, S, DIRECTION, Average)

#define RESNET_BENCHMARK(N, C, W, H, K, S)            \
  RESNET_BM_WITH_DIRECTION(N, C, W, H, K, S, Forward) \
  RESNET_BM_WITH_DIRECTION(N, C, W, H, K, S, Backpropagate)

// Standard benchmark sizes (batch size: 1, 4, optionally 32
#define RESNET_PARAMS(channels, width, height, window, stride) \
  RESNET_BENCHMARK(1, channels, width, height, window, stride);
#include "bench/pooling/resnet_params.def"
#undef RESNET_PARAMS

#define RESNET_PARAMS(channels, width, height, window, stride) \
  RESNET_BENCHMARK(4, channels, width, height, window, stride);
#include "bench/pooling/resnet_params.def"
#undef RESNET_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define RESNET_PARAMS(channels, width, height, window, stride) \
  RESNET_BENCHMARK(32, channels, width, height, window, stride);
#include "bench/pooling/resnet_params.def"
#undef RESNET_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS

// Extended benchmarks (batch size: 2, optionally 8, 16, 64)
#ifdef SNN_EXTENDED_BENCHMARKS
#define RESNET_PARAMS(channels, width, height, window, stride) \
  RESNET_BENCHMARK(2, channels, width, height, window, stride);
#include "bench/pooling/resnet_params.def"
#undef RESNET_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define RESNET_PARAMS(channels, width, height, window, stride) \
  RESNET_BENCHMARK(8, channels, width, height, window, stride);
#include "bench/pooling/resnet_params.def"
#undef RESNET_PARAMS

#define RESNET_PARAMS(channels, width, height, window, stride) \
  RESNET_BENCHMARK(16, channels, width, height, window, stride);
#include "bench/pooling/resnet_params.def"
#undef RESNET_PARAMS

#define RESNET_PARAMS(channels, width, height, window, stride) \
  RESNET_BENCHMARK(64, channels, width, height, window, stride);
#include "bench/pooling/resnet_params.def"
#undef RESNET_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS
#endif  // SNN_EXTENDED_BENCHMARKS
