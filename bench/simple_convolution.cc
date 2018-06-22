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
#include "fixture.h"

#define CONVOLUTION_BENCHMARK(name, ...)                               \
  BENCHMARK_TEMPLATE_DEFINE_F(ConvolutionBenchmark, name, __VA_ARGS__) \
  (benchmark::State & state) { this->run(state); }                     \
  BENCHMARK_REGISTER_F(ConvolutionBenchmark, name)->UseManualTime()

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
}

// Register forward convolution benchmarks..
CONVOLUTION_BENCHMARK(DirectForward, Dense3x3Params,
                      sycldnn::conv2d::conv_type::Forward,
                      sycldnn::conv2d::DirectSelector);
CONVOLUTION_BENCHMARK(TiledForward, Dense3x3Params,
                      sycldnn::conv2d::conv_type::Forward,
                      sycldnn::conv2d::TiledSelector);

CONVOLUTION_BENCHMARK(DirectForwardStride2, Stride2_3x3Params,
                      sycldnn::conv2d::conv_type::Forward,
                      sycldnn::conv2d::DirectSelector);
CONVOLUTION_BENCHMARK(TiledForwardStride2, Stride2_3x3Params,
                      sycldnn::conv2d::conv_type::Forward,
                      sycldnn::conv2d::TiledSelector);

/// Register input back-propagation benchmarks.
CONVOLUTION_BENCHMARK(DirectInputBackprop, Dense3x3Params,
                      sycldnn::conv2d::conv_type::InputBackprop,
                      sycldnn::conv2d::DirectSelector);
CONVOLUTION_BENCHMARK(TiledInputBackprop, Dense3x3Params,
                      sycldnn::conv2d::conv_type::InputBackprop,
                      sycldnn::conv2d::TiledSelector);

CONVOLUTION_BENCHMARK(DirectInputBackpropStride2, Stride2_3x3Params,
                      sycldnn::conv2d::conv_type::InputBackprop,
                      sycldnn::conv2d::DirectSelector);
CONVOLUTION_BENCHMARK(TiledInputBackpropStride2, Stride2_3x3Params,
                      sycldnn::conv2d::conv_type::InputBackprop,
                      sycldnn::conv2d::TiledSelector);

/// Register filter back-propagation benchmarks.
CONVOLUTION_BENCHMARK(DirectFilterBackprop, Dense3x3Params,
                      sycldnn::conv2d::conv_type::FilterBackprop,
                      sycldnn::conv2d::DirectSelector);
CONVOLUTION_BENCHMARK(TiledFilterBackprop, Dense3x3Params,
                      sycldnn::conv2d::conv_type::FilterBackprop,
                      sycldnn::conv2d::TiledSelector);

CONVOLUTION_BENCHMARK(DirectFilterBackpropStride2, Stride2_3x3Params,
                      sycldnn::conv2d::conv_type::FilterBackprop,
                      sycldnn::conv2d::DirectSelector);
CONVOLUTION_BENCHMARK(TiledFilterBackpropStride2, Stride2_3x3Params,
                      sycldnn::conv2d::conv_type::FilterBackprop,
                      sycldnn::conv2d::TiledSelector);
