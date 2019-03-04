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
#include "snn_fixture.h"
#include "ssd_mobilenet_param_set.h"

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

#define SSD_MOBILENET_BENCHMARK_WITH_ALGO_AND_DIR_AND_BACK(                \
    N, Win, Str, Row, Col, Ch, Feat, Pad, Algo, Dir, Back)                 \
  CONVOLUTION_BENCHMARK(                                                   \
      "SSD + MobileNet",                                                   \
      Algo##_##Dir##_##N##_##Win##_##Row##_##Col##_##Ch##_##Feat##_##Back, \
      sycldnn::backend::Back,                                              \
      ParameterSet<N, Win, Str, Row, Col, Ch, Feat, Pad>,                  \
      sycldnn::conv2d::conv_type::Dir, sycldnn::conv2d::Algo##Selector)

#ifdef SNN_BENCH_EIGEN
#define SSD_MOBILENET_BENCHMARK_WITH_EIGEN(N, Win, Str, Row, Col, Ch, Feat, \
                                           Pad, Algo, Dir)                  \
  SSD_MOBILENET_BENCHMARK_WITH_ALGO_AND_DIR_AND_BACK(                       \
      N, Win, Str, Row, Col, Ch, Feat, Pad, Algo, Dir, EigenBackend)
#else
#define SSD_MOBILENET_BENCHMARK_WITH_EIGEN(N, Win, Str, Row, Col, Ch, Feat, \
                                           Pad, Algo, Dir)
#endif

#ifdef SNN_BENCH_SYCLBLAS
#define SSD_MOBILENET_BENCHMARK_WITH_SYCLBLAS(N, Win, Str, Row, Col, Ch, Feat, \
                                              Pad, Algo, Dir)                  \
  SSD_MOBILENET_BENCHMARK_WITH_ALGO_AND_DIR_AND_BACK(                          \
      N, Win, Str, Row, Col, Ch, Feat, Pad, Algo, Dir, SyclBLASBackend)
#else
#define SSD_MOBILENET_BENCHMARK_WITH_SYCLBLAS(N, Win, Str, Row, Col, Ch, Feat, \
                                              Pad, Algo, Dir)
#endif

#define SSD_MOBILENET_BENCHMARK_WITH_ALGO_AND_DIR(N, Win, Str, Row, Col, Ch,  \
                                                  Feat, Pad, Algo, Dir)       \
  SSD_MOBILENET_BENCHMARK_WITH_EIGEN(N, Win, Str, Row, Col, Ch, Feat, Pad,    \
                                     Algo, Dir)                               \
  SSD_MOBILENET_BENCHMARK_WITH_SYCLBLAS(N, Win, Str, Row, Col, Ch, Feat, Pad, \
                                        Algo, Dir)

#define SSD_MOBILENET_BENCHMARK_WITH_ALGO(N, Win, Str, Row, Col, Ch, Feat,   \
                                          Pad, Algo)                         \
  SSD_MOBILENET_BENCHMARK_WITH_ALGO_AND_DIR(N, Win, Str, Row, Col, Ch, Feat, \
                                            Pad, Algo, Forward)              \
  SSD_MOBILENET_BENCHMARK_WITH_ALGO_AND_DIR(N, Win, Str, Row, Col, Ch, Feat, \
                                            Pad, Algo, InputBackprop)        \
  SSD_MOBILENET_BENCHMARK_WITH_ALGO_AND_DIR(N, Win, Str, Row, Col, Ch, Feat, \
                                            Pad, Algo, FilterBackprop)

#define SSD_MOBILENET_BENCHMARK(N, Win, Str, Row, Col, Ch, Feat, Pad)     \
  SSD_MOBILENET_BENCHMARK_WITH_ALGO(N, Win, Str, Row, Col, Ch, Feat, Pad, \
                                    Direct)                               \
  SSD_MOBILENET_BENCHMARK_WITH_ALGO(N, Win, Str, Row, Col, Ch, Feat, Pad, \
                                    Tiled)                                \
  SSD_MOBILENET_BENCHMARK_WITH_ALGO(N, Win, Str, Row, Col, Ch, Feat, Pad, \
                                    Im2col)                               \
  SSD_MOBILENET_BENCHMARK_WITH_ALGO(N, Win, Str, Row, Col, Ch, Feat, Pad, \
                                    Winograd)                             \
  SSD_MOBILENET_BENCHMARK_WITH_ALGO(N, Win, Str, Row, Col, Ch, Feat, Pad, \
                                    Matmul)

// Standard benchmark sizes (batch size: 1, 4, optionally 32
#define SSD_MOBILENET_PARAMS(Win, Str, Row, Col, Ch, Feat, Pad) \
  SSD_MOBILENET_BENCHMARK(1, Win, Str, Row, Col, Ch, Feat, Pad);
#include "bench/conv2d/ssd_mobilenet_params.def"
#undef SSD_MOBILENET_PARAMS

#define SSD_MOBILENET_PARAMS(Win, Str, Row, Col, Ch, Feat, Pad) \
  SSD_MOBILENET_BENCHMARK(4, Win, Str, Row, Col, Ch, Feat, Pad);
#include "bench/conv2d/ssd_mobilenet_params.def"
#undef SSD_MOBILENET_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define SSD_MOBILENET_PARAMS(Win, Str, Row, Col, Ch, Feat, Pad) \
  SSD_MOBILENET_BENCHMARK(32, Win, Str, Row, Col, Ch, Feat, Pad);
#include "bench/conv2d/ssd_mobilenet_params.def"
#undef SSD_MOBILENET_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS

// Extended benchmarks (batch size: 2, optionally 8, 16, 64)
#ifdef SNN_EXTENDED_BENCHMARKS
#define SSD_MOBILENET_PARAMS(Win, Str, Row, Col, Ch, Feat, Pad) \
  SSD_MOBILENET_BENCHMARK(2, Win, Str, Row, Col, Ch, Feat, Pad);
#include "bench/conv2d/ssd_mobilenet_params.def"
#undef SSD_MOBILENET_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define SSD_MOBILENET_PARAMS(Win, Str, Row, Col, Ch, Feat, Pad) \
  SSD_MOBILENET_BENCHMARK(8, Win, Str, Row, Col, Ch, Feat, Pad);
#include "bench/conv2d/ssd_mobilenet_params.def"
#undef SSD_MOBILENET_PARAMS

#define SSD_MOBILENET_PARAMS(Win, Str, Row, Col, Ch, Feat, Pad) \
  SSD_MOBILENET_BENCHMARK(16, Win, Str, Row, Col, Ch, Feat, Pad);
#include "bench/conv2d/ssd_mobilenet_params.def"
#undef SSD_MOBILENET_PARAMS

#define SSD_MOBILENET_PARAMS(Win, Str, Row, Col, Ch, Feat, Pad) \
  SSD_MOBILENET_BENCHMARK(64, Win, Str, Row, Col, Ch, Feat, Pad);
#include "bench/conv2d/ssd_mobilenet_params.def"
#undef SSD_MOBILENET_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS
#endif  // SNN_EXTENDED_BENCHMARKS
