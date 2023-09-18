/*
 * Copyright Codeplay Software Ltd.
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
#ifndef PORTDNN_BENCH_FIXTURE_H_
#define PORTDNN_BENCH_FIXTURE_H_

#include <benchmark/benchmark.h>

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/depthwise_conv2d/params.h"
#include "portdnn/depthwise_conv2d/sizes.h"

extern const char* commit_date;
extern const char* commit_hash;

class BaseDepthwiseConvolutionBenchmark : public benchmark::Fixture {
 private:
  using State = benchmark::State;
  using DepthwiseConv2DParams =
      sycldnn::depthwise_conv2d::DepthwiseConv2DParams;
  using DepthwiseConvSizes = sycldnn::depthwise_conv2d::ConvSizes;

 public:
  // Adds the depthwise convolution parameters to the counter set.
  void add_param_counters(State& state, DepthwiseConv2DParams const& params);

  // Adds theoretical best-case bandwidth requirements to the counter set.
  template <typename T>
  void add_bandwidth_counters(State& state, DepthwiseConvSizes const& sizes);

  // Records the number of elements processed to the counter set. How this is
  // calculated varies based on the type of convolution.
  template <typename ConvType>
  void set_items_processed(State& state, DepthwiseConv2DParams const& params);
};

// Add a full set of counters corresponding to the depthwise convolution
// parameters.
void BaseDepthwiseConvolutionBenchmark::add_param_counters(
    benchmark::State& state, DepthwiseConv2DParams const& params) {
  state.counters["batch"] = params.batch;
  state.counters["in_rows"] = params.in_rows;
  state.counters["in_cols"] = params.in_cols;
  state.counters["channels"] = params.channels;
  state.counters["channel_multiplier"] = params.channel_multiplier;
  state.counters["out_rows"] = params.out_rows;
  state.counters["out_cols"] = params.out_cols;
  state.counters["stride_rows"] = params.stride_rows;
  state.counters["stride_cols"] = params.stride_cols;
  state.counters["fil_rows"] = params.window_rows;
  state.counters["fil_cols"] = params.window_cols;
  state.counters["pad_rows"] = params.pad_rows;
  state.counters["pad_cols"] = params.pad_cols;
}

// Calculate the optimal bandwidth requirements, and add corresponding counters.
// This assumes each filter and input element is read exactly once, rather than
// the actual behaviour where multiple threads may re-read the same values.
template <typename ElementType>
void BaseDepthwiseConvolutionBenchmark::add_bandwidth_counters(
    benchmark::State& state, DepthwiseConvSizes const& sizes) {
  // Compute the size of each element in bytes.
  auto element_bytes = sizeof(ElementType);

  state.counters["bytes_read"] =
      (sizes.filter_size + sizes.input_size) * element_bytes;
  state.counters["bytes_written"] = sizes.output_size * element_bytes;
}

template <>
void BaseDepthwiseConvolutionBenchmark::set_items_processed<
    sycldnn::conv2d::conv_type::Forward>(benchmark::State& state,
                                         DepthwiseConv2DParams const& params) {
  // We require a fused multiply-add for each value in the input with each value
  // in the filter, giving an upper bound on the number of items processed.
  auto window_size = params.window_rows * params.window_cols;
  auto tensor_size = params.batch * params.out_rows * params.out_cols *
                     params.channels * params.channel_multiplier;
  auto num_ops = 2;
  state.SetItemsProcessed(state.iterations() * window_size * tensor_size *
                          num_ops);
}

template <>
void BaseDepthwiseConvolutionBenchmark::set_items_processed<
    sycldnn::conv2d::conv_type::InputBackprop>(
    benchmark::State& state, DepthwiseConv2DParams const& params) {
  // For the backprop steps we perform another convolution, so the only
  // real difference is that the output is the input.
  auto window_size = params.window_rows * params.window_cols;
  auto tensor_size = params.batch * params.in_rows * params.in_cols *
                     params.channels * params.channel_multiplier;
  auto num_ops = 2;
  state.SetItemsProcessed(state.iterations() * window_size * tensor_size *
                          num_ops);
}

template <>
void BaseDepthwiseConvolutionBenchmark::set_items_processed<
    sycldnn::conv2d::conv_type::FilterBackprop>(
    benchmark::State& state, DepthwiseConv2DParams const& params) {
  // We are accumulating the error in the filter, so we perform a convolution
  // over the input with the output.
  auto window_size = params.window_rows * params.window_cols;
  auto tensor_size = params.batch * params.out_rows * params.out_cols *
                     params.channels * params.channel_multiplier;
  auto num_ops = 2;
  state.SetItemsProcessed(state.iterations() * window_size * tensor_size *
                          num_ops);
}

#endif  // define PORTDNN_BENCH_FIXTURE_H_
