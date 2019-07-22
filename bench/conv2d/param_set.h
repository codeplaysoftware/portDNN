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
#ifndef SYCLDNN_BENCH_CONV2D_PARAM_SET_H_
#define SYCLDNN_BENCH_CONV2D_PARAM_SET_H_

#include "sycldnn/padding_mode.h"

#include "sycldnn/conv2d/params.h"

#include "sycldnn/helpers/padding.h"

#include <benchmark/benchmark.h>

#include <vector>

/**
 * Function object which returns a conv2d parameter struct with the given
 * parameters.
 *
 * \tparam Batches Number of batches
 * \tparam Window Size of convolution window
 * \tparam Stride Stride of the convolution
 * \tparam Rows Number of rows in the input
 * \tparam Cols Number of columns in the input
 * \tparam Channels Number of channels
 * \tparam Features Number of features
 */
template <int Batches, int Window, int Stride, int Rows, int Cols, int Channels,
          int Features, sycldnn::PaddingMode Mode>
struct ParameterSet {
  sycldnn::conv2d::Conv2DParams operator()() {
    sycldnn::conv2d::Conv2DParams params;
    params.channels = Channels;
    params.features = Features;
    params.batch = Batches;
    params.in_rows = Rows;
    params.in_cols = Cols;
    params.window_rows = Window;
    params.window_cols = Window;
    params.stride_rows = Stride;
    params.stride_cols = Stride;
    params.dilation_rows = 1;
    params.dilation_cols = 1;
    return sycldnn::helpers::add_padding_to(params, Mode);
  }
};

/**
 * Namespace containing convolution parameter serialization and deserialization
 * routines to allow them to be passed into benchmarks at runtime.
 */
namespace benchmark_params {

/**
 * Encode convolution parameters as a vector.
 *
 * By passing this vector as an argument to a benchmark::internal::Benchmark
 * instance, these parameters can be provided to each benchmark::State for that
 * benchmark.
 */
inline std::vector<int> serialize(int batch, int window, int stride, int rows,
                                  int cols, int channels, int features,
                                  sycldnn::PaddingMode mode) {
  return {batch, window,   stride,   rows,
          cols,  channels, features, static_cast<int>(mode)};
}

/**
 * Extract convolution parameters from a benchmark::State instance.
 *
 * Expects the parameters of the benchmark::State to match those provided by the
 * serialize function.
 */
inline sycldnn::conv2d::Conv2DParams deserialize(
    benchmark::State const& state) {
  sycldnn::conv2d::Conv2DParams params;
  params.batch = state.range(0);
  params.window_rows = state.range(1);
  params.window_cols = state.range(1);
  params.stride_rows = state.range(2);
  params.stride_cols = state.range(2);
  params.in_rows = state.range(3);
  params.in_cols = state.range(4);
  params.channels = state.range(5);
  params.features = state.range(6);
  params.dilation_rows = 1;
  params.dilation_cols = 1;
  auto mode = static_cast<sycldnn::PaddingMode>(state.range(7));
  return sycldnn::helpers::add_padding_to(params, mode);
}

}  // namespace benchmark_params

#endif  // SYCLDNN_BENCH_CONV2D_PARAM_SET_H_
