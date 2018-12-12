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

#define CONVOLUTION_BENCHMARKS_WITH_ALGO_AND_DIR(Algo, Dir)                 \
  CONVOLUTION_BENCHMARK("SimpleConvolution", Algo##Dir, Dense3x3Params,     \
                        sycldnn::conv2d::conv_type::Dir,                    \
                        sycldnn::conv2d::Algo##Selector);                   \
  CONVOLUTION_BENCHMARK("SimpleConvolution", Algo##Dir##Stride2,            \
                        Stride2_3x3Params, sycldnn::conv2d::conv_type::Dir, \
                        sycldnn::conv2d::Algo##Selector)

#define CONVOLUTION_BENCHMARKS_WITH_DIR(Dir)            \
  CONVOLUTION_BENCHMARKS_WITH_ALGO_AND_DIR(Direct, Dir) \
  CONVOLUTION_BENCHMARKS_WITH_ALGO_AND_DIR(Tiled, Dir)  \
  CONVOLUTION_BENCHMARKS_WITH_ALGO_AND_DIR(Im2col, Dir) \
  CONVOLUTION_BENCHMARKS_WITH_ALGO_AND_DIR(Winograd, Dir)

// Register forward convolution benchmarks..
CONVOLUTION_BENCHMARKS_WITH_DIR(Forward);

/// Register input back-propagation benchmarks.
CONVOLUTION_BENCHMARKS_WITH_DIR(InputBackprop);

/// Register filter back-propagation benchmarks.
CONVOLUTION_BENCHMARKS_WITH_DIR(FilterBackprop);
