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

#ifdef SNN_BENCH_EIGEN
#include "src/backend/eigen_backend_provider.h"

#include "sycldnn/backend/eigen_backend.h"
#endif  // SNN_BENCH_EIGEN

#ifdef SNN_BENCH_SYCLBLAS
#include "src/backend/syclblas_backend_provider.h"
#include "sycldnn/backend/sycl_blas_backend.h"
#endif  // SNN_BENCH_SYCLBLAS

#include "sycldnn/padding_mode.h"

#include "sycldnn/conv2d/conv_type.h"
#include "sycldnn/conv2d/params.h"

#include "sycldnn/conv2d/selector/direct_selector.h"
#include "sycldnn/conv2d/selector/im2col_selector.h"
#include "sycldnn/conv2d/selector/tiled_selector.h"
#include "sycldnn/conv2d/selector/winograd_selector.h"

#include <vector>

#if !defined(SNN_BENCH_EIGEN) && !defined(SNN_BENCH_SYCLBLAS)
#error At least one of SNN_BENCH_EIGEN or SNN_BENCH_SYCLBLAS must be set.
#endif

#define CONFIG(N, WIN, STR, H, W, C, F, MOD) \
  benchmark_params::serialize(N, WIN, STR, H, W, C, F, MOD)

std::vector<std::vector<int>> const& get_benchmark_configs() {
  static std::vector<std::vector<int>> const configs = {
      CONFIG(1, 3, 1, 27, 27, 196, 384, sycldnn::PaddingMode::SAME),
      CONFIG(1, 3, 2, 27, 27, 196, 384, sycldnn::PaddingMode::SAME),
  };
  return configs;
}

#define BENCHMARK_WITH_ALGO_AND_DIR_AND_BACK_AND_DTYPE(Algo, Dir, Back, DType) \
  CONVOLUTION_BENCHMARK(                                                       \
      "SimpleConvolution", Algo##Dir##Back, sycldnn::backend::Back, DType,     \
      sycldnn::conv2d::conv_type::Dir, sycldnn::conv2d::Algo##Selector)

#define BENCHMARK_WITH_ALGO_AND_DIR_AND_BACK(Algo, Dir, Back) \
  BENCHMARK_WITH_ALGO_AND_DIR_AND_BACK_AND_DTYPE(Algo, Dir, Back, float)

#ifdef SNN_BENCH_EIGEN
#define BENCHMARK_WITH_EIGEN(Algo, Dir) \
  BENCHMARK_WITH_ALGO_AND_DIR_AND_BACK(Algo, Dir, EigenBackend)
#else
#define BENCHMARK_WITH_EIGEN(Algo, Dir)
#endif

#ifdef SNN_BENCH_SYCLBLAS
#define BENCHMARK_WITH_SYCLBLAS(Algo, Dir) \
  BENCHMARK_WITH_ALGO_AND_DIR_AND_BACK(Algo, Dir, SyclBLASBackend)
#else
#define BENCHMARK_WITH_SYCLBLAS(Algo, Dir)
#endif

#define BENCHMARKS_WITH_ALGO_AND_DIR(Algo, Dir) \
  BENCHMARK_WITH_EIGEN(Algo, Dir)               \
  BENCHMARK_WITH_SYCLBLAS(Algo, Dir)

#define BENCHMARKS_WITH_DIR(Dir)            \
  BENCHMARKS_WITH_ALGO_AND_DIR(Direct, Dir) \
  BENCHMARKS_WITH_ALGO_AND_DIR(Tiled, Dir)  \
  BENCHMARKS_WITH_ALGO_AND_DIR(Im2col, Dir) \
  BENCHMARKS_WITH_ALGO_AND_DIR(Winograd, Dir)

// Register forward convolution benchmarks.
BENCHMARKS_WITH_DIR(Forward);

// Register input back-propagation benchmarks.
BENCHMARKS_WITH_DIR(InputBackprop);

// Register filter back-propagation benchmarks.
BENCHMARKS_WITH_DIR(FilterBackprop);
