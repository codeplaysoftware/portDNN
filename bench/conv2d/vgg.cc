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

#ifdef SNN_BENCH_EIGEN
#include "src/backend/eigen_backend_provider.h"

#include "sycldnn/backend/eigen_backend.h"
#endif  // SNN_BENCH_EIGEN

#ifdef SNN_BENCH_SYCLBLAS
#include "src/backend/syclblas_backend_provider.h"
#include "sycldnn/backend/sycl_blas_backend.h"
#endif  // SNN_BENCH_SYCLBLAS

#include "sycldnn/conv2d/conv_type.h"

#include "sycldnn/conv2d/selector/direct_selector.h"
#include "sycldnn/conv2d/selector/im2col_selector.h"
#include "sycldnn/conv2d/selector/matmul_selector.h"
#include "sycldnn/conv2d/selector/tiled_selector.h"
#include "sycldnn/conv2d/selector/winograd_selector.h"

#if !defined(SNN_BENCH_EIGEN) && !defined(SNN_BENCH_SYCLBLAS)
#error At least one of SNN_BENCH_EIGEN or SNN_BENCH_SYCLBLAS must be set.
#endif

#define VGG_BENCHMARK_WITH_ALGO_AND_DIR_AND_BACK(N, C, W, H, F, Algo, Dir, \
                                                 Back)                     \
  CONVOLUTION_BENCHMARK(                                                   \
      "VGG", Algo##_##Dir##_##N##_##C##_##W##_##H##_##F##_##Back,          \
      sycldnn::backend::Back, ParameterSet<N, C, W, H, F>,                 \
      sycldnn::conv2d::conv_type::Dir, sycldnn::conv2d::Algo##Selector)

#ifdef SNN_BENCH_EIGEN
#define VGG_BENCHMARK_WITH_EIGEN(N, C, W, H, F, Algo, Dir)           \
  VGG_BENCHMARK_WITH_ALGO_AND_DIR_AND_BACK(N, C, W, H, F, Algo, Dir, \
                                           EigenBackend)
#else
#define VGG_BENCHMARK_WITH_EIGEN(N, C, W, H, F, Algo, Dir)
#endif

#ifdef SNN_BENCH_SYCLBLAS
#define VGG_BENCHMARK_WITH_SYCLBLAS(N, C, W, H, F, Algo, Dir)        \
  VGG_BENCHMARK_WITH_ALGO_AND_DIR_AND_BACK(N, C, W, H, F, Algo, Dir, \
                                           SyclBLASBackend)
#else
#define VGG_BENCHMARK_WITH_SYCLBLAS(N, C, W, H, F, Algo, Dir)
#endif

#define VGG_BENCHMARK_WITH_ALGO_AND_DIR(N, C, W, H, F, Algo, Dir) \
  VGG_BENCHMARK_WITH_EIGEN(N, C, W, H, F, Algo, Dir)              \
  VGG_BENCHMARK_WITH_SYCLBLAS(N, C, W, H, F, Algo, Dir)

#define VGG_BENCHMARK_WITH_ALGO(N, C, W, H, F, Algo)                  \
  VGG_BENCHMARK_WITH_ALGO_AND_DIR(N, C, W, H, F, Algo, Forward)       \
  VGG_BENCHMARK_WITH_ALGO_AND_DIR(N, C, W, H, F, Algo, InputBackprop) \
  VGG_BENCHMARK_WITH_ALGO_AND_DIR(N, C, W, H, F, Algo, FilterBackprop)

#define VGG_BENCHMARK(N, C, W, H, F)               \
  VGG_BENCHMARK_WITH_ALGO(N, C, W, H, F, Direct)   \
  VGG_BENCHMARK_WITH_ALGO(N, C, W, H, F, Tiled)    \
  VGG_BENCHMARK_WITH_ALGO(N, C, W, H, F, Im2col)   \
  VGG_BENCHMARK_WITH_ALGO(N, C, W, H, F, Winograd) \
  VGG_BENCHMARK_WITH_ALGO(N, C, W, H, F, WinogradLarge)

// Standard benchmark sizes (batch size: 1, 4, optionally 32
#define VGG_PARAMS(channels, width, height, features) \
  VGG_BENCHMARK(1, channels, width, height, features);
#include "bench/conv2d/vgg_params.def"
#undef VGG_PARAMS

#define VGG_PARAMS(channels, width, height, features) \
  VGG_BENCHMARK(4, channels, width, height, features);
#include "bench/conv2d/vgg_params.def"
#undef VGG_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define VGG_PARAMS(channels, width, height, features) \
  VGG_BENCHMARK(32, channels, width, height, features);
#include "bench/conv2d/vgg_params.def"
#undef VGG_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS

// Extended benchmarks (batch size: 2, optionally 8, 16, 64)
#ifdef SNN_EXTENDED_BENCHMARKS
#define VGG_PARAMS(channels, width, height, features) \
  VGG_BENCHMARK(2, channels, width, height, features);
#include "bench/conv2d/vgg_params.def"
#undef VGG_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define VGG_PARAMS(channels, width, height, features) \
  VGG_BENCHMARK(8, channels, width, height, features);
#include "bench/conv2d/vgg_params.def"
#undef VGG_PARAMS

#define VGG_PARAMS(channels, width, height, features) \
  VGG_BENCHMARK(16, channels, width, height, features);
#include "bench/conv2d/vgg_params.def"
#undef VGG_PARAMS

#define VGG_PARAMS(channels, width, height, features) \
  VGG_BENCHMARK(64, channels, width, height, features);
#include "bench/conv2d/vgg_params.def"
#undef VGG_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS
#endif  // SNN_EXTENDED_BENCHMARKS
