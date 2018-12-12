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

#include "sycldnn/conv2d/selector/direct_selector.h"
#include "sycldnn/conv2d/selector/im2col_selector.h"
#include "sycldnn/conv2d/selector/matmul_selector.h"
#include "sycldnn/conv2d/selector/tiled_selector.h"
#include "sycldnn/conv2d/selector/winograd_selector.h"

#define RESNET_BENCHMARK_WITH_ALGO_AND_DIR(N, C, W, H, Flt, S, Ftr, Algo, Dir) \
  CONVOLUTION_BENCHMARK(                                                       \
      "ResNet", Algo##_##Dir##_##N##_##C##_##W##_##H##_##Flt##_##S##_##Ftr,    \
      ParameterSet<N, C, W, H, Flt, S, Ftr>, sycldnn::conv2d::conv_type::Dir,  \
      sycldnn::conv2d::Algo##Selector)

#define RESNET_BENCHMARK_WITH_ALGO(N, C, W, H, Flt, S, Ftr, Algo)            \
  RESNET_BENCHMARK_WITH_ALGO_AND_DIR(N, C, W, H, Flt, S, Ftr, Algo, Forward) \
  RESNET_BENCHMARK_WITH_ALGO_AND_DIR(N, C, W, H, Flt, S, Ftr, Algo,          \
                                     InputBackprop)                          \
  RESNET_BENCHMARK_WITH_ALGO_AND_DIR(N, C, W, H, Flt, S, Ftr, Algo,          \
                                     FilterBackprop)

#define RESNET_BENCHMARK(N, C, W, H, Flt, S, Ftr)               \
  RESNET_BENCHMARK_WITH_ALGO(N, C, W, H, Flt, S, Ftr, Direct)   \
  RESNET_BENCHMARK_WITH_ALGO(N, C, W, H, Flt, S, Ftr, Tiled)    \
  RESNET_BENCHMARK_WITH_ALGO(N, C, W, H, Flt, S, Ftr, Im2col)   \
  RESNET_BENCHMARK_WITH_ALGO(N, C, W, H, Flt, S, Ftr, Winograd) \
  RESNET_BENCHMARK_WITH_ALGO(N, C, W, H, Flt, S, Ftr, Matmul)

// Standard benchmark sizes (batch size: 1, 4, optionally 32
#define RESNET_PARAMS(channels, width, height, filter, stride, features) \
  RESNET_BENCHMARK(1, channels, width, height, filter, stride, features);
#include "bench/conv2d/resnet_params.def"
#undef RESNET_PARAMS

#define RESNET_PARAMS(channels, width, height, filter, stride, features) \
  RESNET_BENCHMARK(4, channels, width, height, filter, stride, features);
#include "bench/conv2d/resnet_params.def"
#undef RESNET_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define RESNET_PARAMS(channels, width, height, filter, stride, features) \
  RESNET_BENCHMARK(32, channels, width, height, filter, stride, features);
#include "bench/conv2d/resnet_params.def"
#undef RESNET_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS

// Extended benchmarks (batch size: 2, optionally 8, 16, 64)
#ifdef SNN_EXTENDED_BENCHMARKS
#define RESNET_PARAMS(channels, width, height, filter, stride, features) \
  RESNET_BENCHMARK(2, channels, width, height, filter, stride, features);
#include "bench/conv2d/resnet_params.def"
#undef RESNET_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define RESNET_PARAMS(channels, width, height, filter, stride, features) \
  RESNET_BENCHMARK(8, channels, width, height, filter, stride, features);
#include "bench/conv2d/resnet_params.def"
#undef RESNET_PARAMS

#define RESNET_PARAMS(channels, width, height, filter, stride, features) \
  RESNET_BENCHMARK(16, channels, width, height, filter, stride, features);
#include "bench/conv2d/resnet_params.def"
#undef RESNET_PARAMS

#define RESNET_PARAMS(channels, width, height, filter, stride, features) \
  RESNET_BENCHMARK(64, channels, width, height, filter, stride, features);
#include "bench/conv2d/resnet_params.def"
#undef RESNET_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS
#endif  // SNN_EXTENDED_BENCHMARKS
