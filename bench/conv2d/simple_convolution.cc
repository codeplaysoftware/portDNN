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

#include "sycldnn/conv2d/conv_type.h"
#include "sycldnn/conv2d/params.h"

#include "sycldnn/conv2d/selector/direct_selector.h"
#include "sycldnn/conv2d/selector/im2col_selector.h"
#include "sycldnn/conv2d/selector/tiled_selector.h"
#include "sycldnn/conv2d/selector/winograd_selector.h"

namespace {

struct Dense3x3Params {
  sycldnn::conv2d::Conv2DParams operator()() {
    sycldnn::conv2d::Conv2DParams params;
    params.channels = 196;
    params.features = 384;
    params.batch = 16;
    params.in_rows = 27;
    params.in_cols = 27;
    params.window_rows = 3;
    params.window_cols = 3;
    params.stride_rows = 1;
    params.stride_cols = 1;
    params.out_rows = 27;
    params.out_cols = 27;
    params.pad_rows = 1;
    params.pad_cols = 1;
    params.dilation_rows = 1;
    params.dilation_cols = 1;
    return params;
  }
};

struct Stride2_3x3Params {
  sycldnn::conv2d::Conv2DParams operator()() {
    sycldnn::conv2d::Conv2DParams params;
    params.channels = 196;
    params.features = 384;
    params.batch = 1;
    params.in_rows = 27;
    params.in_cols = 27;
    params.window_rows = 3;
    params.window_cols = 3;
    params.stride_rows = 2;
    params.stride_cols = 2;
    params.out_rows = 13;
    params.out_cols = 13;
    params.pad_rows = 0;
    params.pad_cols = 0;
    params.dilation_rows = 1;
    params.dilation_cols = 1;
    return params;
  }
};
}  // namespace

#if !defined(SNN_BENCH_EIGEN) && !defined(SNN_BENCH_SYCLBLAS)
#error At least one of SNN_BENCH_EIGEN or SNN_BENCH_SYCLBLAS must be set.
#endif

#define BENCHMARK_WITH_ALGO_AND_DIR_AND_BACK_AND_DTYPE(Algo, Dir, Back, DType) \
  CONVOLUTION_BENCHMARK("SimpleConvolution", Algo##Dir##Back,                  \
                        sycldnn::backend::Back, DType, Dense3x3Params,         \
                        sycldnn::conv2d::conv_type::Dir,                       \
                        sycldnn::conv2d::Algo##Selector);                      \
  CONVOLUTION_BENCHMARK("SimpleConvolution", Algo##Dir##Stride2##Back,         \
                        sycldnn::backend::Back, DType, Stride2_3x3Params,      \
                        sycldnn::conv2d::conv_type::Dir,                       \
                        sycldnn::conv2d::Algo##Selector)

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
