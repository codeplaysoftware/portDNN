/*
 * Copyright Codeplay Software Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
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
#ifndef PORTDNN_BENCH_DEPTHWISE_CONV2D_BENCHMARK_PARAMS_H_
#define PORTDNN_BENCH_DEPTHWISE_CONV2D_BENCHMARK_PARAMS_H_

#include "portdnn/padding_mode.h"

#include "portdnn/depthwise_conv2d/params.h"

#include "portdnn/helpers/padding.h"

#include <benchmark/benchmark.h>

#include <vector>

/**
 * Namespace containing depthwise convolution parameter serialization and
 * deserialization routines to allow them to be passed into benchmarks at
 * runtime.
 */
namespace benchmark_params {

/**
 * Encode depthwise convolution parameters as a vector.
 *
 * By passing this vector as an argument to a benchmark::internal::Benchmark
 * instance, these parameters can be provided to each benchmark::State for that
 * benchmark.
 */
inline std::vector<int> serialize(int batch, int window, int stride, int rows,
                                  int cols, int channels, int multiplier,
                                  sycldnn::PaddingMode mode) {
  return {batch, window,   stride,     rows,
          cols,  channels, multiplier, static_cast<int>(mode)};
}

/**
 * Extract depthwise convolution parameters from a benchmark::State instance.
 *
 * Expects the parameters of the benchmark::State to match those provided by the
 * serialize function.
 */
inline sycldnn::depthwise_conv2d::DepthwiseConv2DParams deserialize(
    benchmark::State const& state) {
  sycldnn::depthwise_conv2d::DepthwiseConv2DParams params;
  params.batch = state.range(0);
  params.window_rows = state.range(1);
  params.window_cols = state.range(1);
  params.stride_rows = state.range(2);
  params.stride_cols = state.range(2);
  params.in_rows = state.range(3);
  params.in_cols = state.range(4);
  params.channels = state.range(5);
  params.channel_multiplier = state.range(6);
  auto mode = static_cast<sycldnn::PaddingMode>(state.range(7));
  return sycldnn::helpers::add_padding_to(params, mode);
}

}  // namespace benchmark_params

#endif  // PORTDNN_BENCH_DEPTHWISE_CONV2D_BENCHMARK_PARAMS_H_
