/*
 * Copyright 2019 Codeplay Software Ltd.
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

#define XCEPTION_BENCHMARK_WITH_ALGO_DIR_BACK_DTYPE(                       \
    N, WIN, STR, H, W, C, F, MOD, Algo, Dir, Back, DType)                  \
  CONVOLUTION_BENCHMARK(                                                   \
      "Xception",                                                          \
      Algo##_##Dir##_##N##_##WIN##_##STR##_##H##_##W##_##C##_##F##_##Back, \
      sycldnn::backend::Back, DType,                                       \
      ParameterSet<N, WIN, STR, H, W, C, F, MOD>,                          \
      sycldnn::conv2d::conv_type::Dir, sycldnn::conv2d::Algo##Selector)

#define XCEPTION_BENCHMARK_WITH_ALGO_DIR_BACK(N, WIN, STR, H, W, C, F, MOD, \
                                              Algo, Dir, Back)              \
  XCEPTION_BENCHMARK_WITH_ALGO_DIR_BACK_DTYPE(N, WIN, STR, H, W, C, F, MOD, \
                                              Algo, Dir, Back, float)

#ifdef SNN_BENCH_EIGEN
#define XCEPTION_BENCHMARK_WITH_EIGEN(N, WIN, STR, H, W, C, F, MOD, Algo, Dir) \
  XCEPTION_BENCHMARK_WITH_ALGO_DIR_BACK(N, WIN, STR, H, W, C, F, MOD, Algo,    \
                                        Dir, EigenBackend)
#else
#define XCEPTION_BENCHMARK_WITH_EIGEN(N, WIN, STR, H, W, C, F, MOD, Algo, Dir)
#endif

#ifdef SNN_BENCH_SYCLBLAS
#define XCEPTION_BENCHMARK_WITH_SYCLBLAS(N, WIN, STR, H, W, C, F, MOD, Algo, \
                                         Dir)                                \
  XCEPTION_BENCHMARK_WITH_ALGO_DIR_BACK(N, WIN, STR, H, W, C, F, MOD, Algo,  \
                                        Dir, SyclBLASBackend)
#else
#define XCEPTION_BENCHMARK_WITH_SYCLBLAS(N, WIN, STR, H, W, C, F, MOD, Algo, \
                                         Dir)
#endif

#define XCEPTION_BENCHMARK_WITH_ALGO_AND_DIR(N, WIN, STR, H, W, C, F, MOD, \
                                             Algo, Dir)                    \
  XCEPTION_BENCHMARK_WITH_EIGEN(N, WIN, STR, H, W, C, F, MOD, Algo, Dir)   \
  XCEPTION_BENCHMARK_WITH_SYCLBLAS(N, WIN, STR, H, W, C, F, MOD, Algo, Dir)

#define XCEPTION_BENCHMARK_WITH_ALGO(N, WIN, STR, H, W, C, F, MOD, Algo)   \
  XCEPTION_BENCHMARK_WITH_ALGO_AND_DIR(N, WIN, STR, H, W, C, F, MOD, Algo, \
                                       Forward)                            \
  XCEPTION_BENCHMARK_WITH_ALGO_AND_DIR(N, WIN, STR, H, W, C, F, MOD, Algo, \
                                       InputBackprop)                      \
  XCEPTION_BENCHMARK_WITH_ALGO_AND_DIR(N, WIN, STR, H, W, C, F, MOD, Algo, \
                                       FilterBackprop)

#define XCEPTION_BENCHMARK(N, WIN, STR, H, W, C, F, MOD)                    \
  XCEPTION_BENCHMARK_WITH_ALGO(N, WIN, STR, H, W, C, F, MOD, Direct)        \
  XCEPTION_BENCHMARK_WITH_ALGO(N, WIN, STR, H, W, C, F, MOD, Tiled)         \
  XCEPTION_BENCHMARK_WITH_ALGO(N, WIN, STR, H, W, C, F, MOD, Im2col)        \
  XCEPTION_BENCHMARK_WITH_ALGO(N, WIN, STR, H, W, C, F, MOD, Winograd)      \
  XCEPTION_BENCHMARK_WITH_ALGO(N, WIN, STR, H, W, C, F, MOD, WinogradLarge) \
  XCEPTION_BENCHMARK_WITH_ALGO(N, WIN, STR, H, W, C, F, MOD, Matmul)

// Standard benchmark sizes (batch size: 1, 4, optionally 32)
#define XCEPTION_PARAMS(WIN, STR, H, W, C, F, MOD) \
  XCEPTION_BENCHMARK(1, WIN, STR, H, W, C, F, MOD);
#include "bench/conv2d/xception_params.def"
#undef XCEPTION_PARAMS

#define XCEPTION_PARAMS(WIN, STR, H, W, C, F, MOD) \
  XCEPTION_BENCHMARK(4, WIN, STR, H, W, C, F, MOD);
#include "bench/conv2d/xception_params.def"
#undef XCEPTION_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define XCEPTION_PARAMS(WIN, STR, H, W, C, F, MOD) \
  XCEPTION_BENCHMARK(32, WIN, STR, H, W, C, F, MOD);
#include "bench/conv2d/xception_params.def"
#undef XCEPTION_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS

// Extended benchmarks (batch size: 2, optionally 8, 16, 64)
#ifdef SNN_EXTENDED_BENCHMARKS
#define XCEPTION_PARAMS(WIN, STR, H, W, C, F, MOD) \
  XCEPTION_BENCHMARK(2, WIN, STR, H, W, C, F, MOD);
#include "bench/conv2d/xception_params.def"
#undef XCEPTION_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define XCEPTION_PARAMS(WIN, STR, H, W, C, F, MOD) \
  XCEPTION_BENCHMARK(8, WIN, STR, H, W, C, F, MOD);
#include "bench/conv2d/xception_params.def"
#undef XCEPTION_PARAMS

#define XCEPTION_PARAMS(WIN, STR, H, W, C, F, MOD) \
  XCEPTION_BENCHMARK(16, WIN, STR, H, W, C, F, MOD);
#include "bench/conv2d/xception_params.def"
#undef XCEPTION_PARAMS

#define XCEPTION_PARAMS(WIN, STR, H, W, C, F, MOD) \
  XCEPTION_BENCHMARK(64, WIN, STR, H, W, C, F, MOD);
#include "bench/conv2d/xception_params.def"
#undef XCEPTION_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS
#endif  // SNN_EXTENDED_BENCHMARKS
