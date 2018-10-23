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
#include "snn_fixture.h"
#include "vgg_param_set.h"

#define VGG_BM_WITH_DIR_AND_OP(N, C, W, H, DIRECTION, OP)                     \
  POOLING_BENCHMARK("VGG", OP##_##DIRECTION##_##N##_##C##_##W##_##H##_2,      \
                    ParameterSet<N, C, W, H, 2>, sycldnn::pooling::DIRECTION, \
                    sycldnn::pooling::OP)

#define VGG_BM_WITH_DIRECTION(N, C, W, H, DIRECTION) \
  VGG_BM_WITH_DIR_AND_OP(N, C, W, H, DIRECTION, Max) \
  VGG_BM_WITH_DIR_AND_OP(N, C, W, H, DIRECTION, Average)

#define VGG_BENCHMARK(N, C, W, H)            \
  VGG_BM_WITH_DIRECTION(N, C, W, H, Forward) \
  VGG_BM_WITH_DIRECTION(N, C, W, H, Backpropagate)

// Standard benchmark sizes (batch size: 1, 4, optionally 32
#define VGG_PARAMS(channels, width, height) \
  VGG_BENCHMARK(1, channels, width, height);
#include "bench/pooling/vgg_params.def"
#undef VGG_PARAMS

#define VGG_PARAMS(channels, width, height) \
  VGG_BENCHMARK(4, channels, width, height);
#include "bench/pooling/vgg_params.def"
#undef VGG_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define VGG_PARAMS(channels, width, height) \
  VGG_BENCHMARK(32, channels, width, height);
#include "bench/pooling/vgg_params.def"
#undef VGG_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS

// Extended benchmarks (batch size: 2, optionally 8, 16, 64)
#ifdef SNN_EXTENDED_BENCHMARKS
#define VGG_PARAMS(channels, width, height) \
  VGG_BENCHMARK(2, channels, width, height);
#include "bench/pooling/vgg_params.def"
#undef VGG_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define VGG_PARAMS(channels, width, height) \
  VGG_BENCHMARK(8, channels, width, height);
#include "bench/pooling/vgg_params.def"
#undef VGG_PARAMS

#define VGG_PARAMS(channels, width, height) \
  VGG_BENCHMARK(16, channels, width, height);
#include "bench/pooling/vgg_params.def"
#undef VGG_PARAMS

#define VGG_PARAMS(channels, width, height) \
  VGG_BENCHMARK(64, channels, width, height);
#include "bench/pooling/vgg_params.def"
#undef VGG_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS
#endif  // SNN_EXTENDED_BENCHMARKS
