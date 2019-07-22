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

#include <vector>

#if !defined(SNN_BENCH_EIGEN) && !defined(SNN_BENCH_SYCLBLAS)
#error At least one of SNN_BENCH_EIGEN or SNN_BENCH_SYCLBLAS must be set.
#endif

#define BM_WITH_ALGO_DIR_BACK_DTYPE(Algo, Dir, Back, DType)             \
  CONVOLUTION_BENCHMARK(                                                \
      "Xception", Algo##_##Dir##_##Back, sycldnn::backend::Back, DType, \
      sycldnn::conv2d::conv_type::Dir, sycldnn::conv2d::Algo##Selector)

#define BM_WITH_ALGO_DIR_BACK(Algo, Dir, Back) \
  BM_WITH_ALGO_DIR_BACK_DTYPE(Algo, Dir, Back, float)

#ifdef SNN_BENCH_EIGEN
#define BM_WITH_EIGEN(Algo, Dir) BM_WITH_ALGO_DIR_BACK(Algo, Dir, EigenBackend)
#else
#define BM_WITH_EIGEN(Algo, Dir)
#endif

#ifdef SNN_BENCH_SYCLBLAS
#define BM_WITH_SYCLBLAS(Algo, Dir) \
  BM_WITH_ALGO_DIR_BACK(Algo, Dir, SyclBLASBackend)
#else
#define BM_WITH_SYCLBLAS(Algo, Dir)
#endif

#define BM_WITH_ALGO_AND_DIR(Algo, Dir) \
  BM_WITH_EIGEN(Algo, Dir)              \
  BM_WITH_SYCLBLAS(Algo, Dir)

#define BM_WITH_ALGO(Algo)                  \
  BM_WITH_ALGO_AND_DIR(Algo, Forward)       \
  BM_WITH_ALGO_AND_DIR(Algo, InputBackprop) \
  BM_WITH_ALGO_AND_DIR(Algo, FilterBackprop)

BM_WITH_ALGO(Direct);
BM_WITH_ALGO(Tiled);
BM_WITH_ALGO(Im2col);
BM_WITH_ALGO(Winograd);
BM_WITH_ALGO(WinogradLarge);
BM_WITH_ALGO(Matmul);

#define CONFIG(N, WIN, STR, H, W, C, F, MOD) \
  benchmark_params::serialize(N, WIN, STR, H, W, C, F, MOD)

std::vector<std::vector<int>> const& get_benchmark_configs() {
  static std::vector<std::vector<int>> const configs = {

// Standard benchmark sizes (batch size: 1, 4, optionally 32)
#define XCEPTION_PARAMS(WIN, STR, H, W, C, F, MOD) \
  CONFIG(1, WIN, STR, H, W, C, F, MOD),
#include "bench/conv2d/xception_params.def"
#undef XCEPTION_PARAMS

#define XCEPTION_PARAMS(WIN, STR, H, W, C, F, MOD) \
  CONFIG(4, WIN, STR, H, W, C, F, MOD),
#include "bench/conv2d/xception_params.def"
#undef XCEPTION_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define XCEPTION_PARAMS(WIN, STR, H, W, C, F, MOD) \
  CONFIG(32, WIN, STR, H, W, C, F, MOD),
#include "bench/conv2d/xception_params.def"
#undef XCEPTION_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS

// Extended benchmarks (batch size: 2, optionally 8, 16, 64)
#ifdef SNN_EXTENDED_BENCHMARKS
#define XCEPTION_PARAMS(WIN, STR, H, W, C, F, MOD) \
  CONFIG(2, WIN, STR, H, W, C, F, MOD),
#include "bench/conv2d/xception_params.def"
#undef XCEPTION_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define XCEPTION_PARAMS(WIN, STR, H, W, C, F, MOD) \
  CONFIG(8, WIN, STR, H, W, C, F, MOD),
#include "bench/conv2d/xception_params.def"
#undef XCEPTION_PARAMS

#define XCEPTION_PARAMS(WIN, STR, H, W, C, F, MOD) \
  CONFIG(16, WIN, STR, H, W, C, F, MOD),
#include "bench/conv2d/xception_params.def"
#undef XCEPTION_PARAMS

#define XCEPTION_PARAMS(WIN, STR, H, W, C, F, MOD) \
  CONFIG(64, WIN, STR, H, W, C, F, MOD),
#include "bench/conv2d/xception_params.def"
#undef XCEPTION_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS
#endif  // SNN_EXTENDED_BENCHMARKS

  };
  return configs;
}
