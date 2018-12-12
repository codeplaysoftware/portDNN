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

#include "sycldnn/conv2d/selector/direct_selector.h"
#include "sycldnn/conv2d/selector/im2col_selector.h"
#include "sycldnn/conv2d/selector/matmul_selector.h"
#include "sycldnn/conv2d/selector/tiled_selector.h"
#include "sycldnn/conv2d/selector/winograd_selector.h"

#if !defined(SNN_BENCH_EIGEN) && !defined(SNN_BENCH_SYCLBLAS)
#error At least one of SNN_BENCH_EIGEN or SNN_BENCH_SYCLBLAS must be set.
#endif

#define MOBILENET_BENCHMARK_WITH_ALGO_AND_DIR_AND_BACK(                      \
    N, Win, Str, Row, Col, Ch, Feat, Algo, Dir, Back)                        \
  CONVOLUTION_BENCHMARK(                                                     \
      "MobileNet",                                                           \
      Algo##_##Dir##_##N##_##Win##_##Row##_##Col##_##Ch##_##Feat##_##Back,   \
      sycldnn::backend::Back, ParameterSet<N, Win, Str, Row, Col, Ch, Feat>, \
      sycldnn::conv2d::conv_type::Dir, sycldnn::conv2d::Algo##Selector)

#ifdef SNN_BENCH_EIGEN
#define MOBILENET_BENCHMARK_WITH_EIGEN(N, Win, Str, Row, Col, Ch, Feat, Algo, \
                                       Dir)                                   \
  MOBILENET_BENCHMARK_WITH_ALGO_AND_DIR_AND_BACK(                             \
      N, Win, Str, Row, Col, Ch, Feat, Algo, Dir, EigenBackend)
#else
#define MOBILENET_BENCHMARK_WITH_EIGEN(N, Win, Str, Row, Col, Ch, Feat, Algo, \
                                       Dir)
#endif

#ifdef SNN_BENCH_SYCLBLAS
#define MOBILENET_BENCHMARK_WITH_SYCLBLAS(N, Win, Str, Row, Col, Ch, Feat, \
                                          Algo, Dir)                       \
  MOBILENET_BENCHMARK_WITH_ALGO_AND_DIR_AND_BACK(                          \
      N, Win, Str, Row, Col, Ch, Feat, Algo, Dir, SyclBLASBackend)
#else
#define MOBILENET_BENCHMARK_WITH_SYCLBLAS(N, Win, Str, Row, Col, Ch, Feat, \
                                          Algo, Dir)
#endif

#define MOBILENET_BENCHMARK_WITH_ALGO_AND_DIR(N, Win, Str, Row, Col, Ch, Feat, \
                                              Algo, Dir)                       \
  MOBILENET_BENCHMARK_WITH_EIGEN(N, Win, Str, Row, Col, Ch, Feat, Algo, Dir)   \
  MOBILENET_BENCHMARK_WITH_SYCLBLAS(N, Win, Str, Row, Col, Ch, Feat, Algo, Dir)

#define MOBILENET_BENCHMARK_WITH_ALGO(N, Win, Str, Row, Col, Ch, Feat, Algo)   \
  MOBILENET_BENCHMARK_WITH_ALGO_AND_DIR(N, Win, Str, Row, Col, Ch, Feat, Algo, \
                                        Forward)                               \
  MOBILENET_BENCHMARK_WITH_ALGO_AND_DIR(N, Win, Str, Row, Col, Ch, Feat, Algo, \
                                        InputBackprop)                         \
  MOBILENET_BENCHMARK_WITH_ALGO_AND_DIR(N, Win, Str, Row, Col, Ch, Feat, Algo, \
                                        FilterBackprop)

#define MOBILENET_BENCHMARK(N, Win, Str, Row, Col, Ch, Feat)             \
  MOBILENET_BENCHMARK_WITH_ALGO(N, Win, Str, Row, Col, Ch, Feat, Direct) \
  MOBILENET_BENCHMARK_WITH_ALGO(N, Win, Str, Row, Col, Ch, Feat, Tiled)  \
  MOBILENET_BENCHMARK_WITH_ALGO(N, Win, Str, Row, Col, Ch, Feat, Im2col) \
  MOBILENET_BENCHMARK_WITH_ALGO(N, Win, Str, Row, Col, Ch, Feat, Winograd)

// Standard benchmark sizes (batch size: 1, 4, optionally 32
#define MOBILENET_PARAMS(Win, Str, Row, Col, Ch, Feat) \
  MOBILENET_BENCHMARK(1, Win, Str, Row, Col, Ch, Feat);
#include "bench/conv2d/mobilenet_params.def"
#undef MOBILENET_PARAMS

#define MOBILENET_PARAMS(Win, Str, Row, Col, Ch, Feat) \
  MOBILENET_BENCHMARK(4, Win, Str, Row, Col, Ch, Feat);
#include "bench/conv2d/mobilenet_params.def"
#undef MOBILENET_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define MOBILENET_PARAMS(Win, Str, Row, Col, Ch, Feat) \
  MOBILENET_BENCHMARK(32, Win, Str, Row, Col, Ch, Feat);
#include "bench/conv2d/mobilenet_params.def"
#undef MOBILENET_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS

// Extended benchmarks (batch size: 2, optionally 8, 16, 64)
#ifdef SNN_EXTENDED_BENCHMARKS
#define MOBILENET_PARAMS(Win, Str, Row, Col, Ch, Feat) \
  MOBILENET_BENCHMARK(2, Win, Str, Row, Col, Ch, Feat);
#include "bench/conv2d/mobilenet_params.def"
#undef MOBILENET_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define MOBILENET_PARAMS(Win, Str, Row, Col, Ch, Feat) \
  MOBILENET_BENCHMARK(8, Win, Str, Row, Col, Ch, Feat);
#include "bench/conv2d/mobilenet_params.def"
#undef MOBILENET_PARAMS

#define MOBILENET_PARAMS(Win, Str, Row, Col, Ch, Feat) \
  MOBILENET_BENCHMARK(16, Win, Str, Row, Col, Ch, Feat);
#include "bench/conv2d/mobilenet_params.def"
#undef MOBILENET_PARAMS

#define MOBILENET_PARAMS(Win, Str, Row, Col, Ch, Feat) \
  MOBILENET_BENCHMARK(32, Win, Str, Row, Col, Ch, Feat);
#include "bench/conv2d/mobilenet_params.def"
#undef MOBILENET_PARAMS

#define MOBILENET_PARAMS(Win, Str, Row, Col, Ch, Feat) \
  MOBILENET_BENCHMARK(64, Win, Str, Row, Col, Ch, Feat);
#include "bench/conv2d/mobilenet_params.def"
#undef MOBILENET_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS
#endif  // SNN_EXTENDED_BENCHMARKS
