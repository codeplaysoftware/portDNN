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

#include "portdnn/conv2d/params.h"
#include "portdnn/conv2d/sizes.h"

extern const char* commit_date;
extern const char* commit_hash;

class BaseConvolutionBenchmark : public benchmark::Fixture {
 private:
  using State = benchmark::State;
  using Conv2DParams = sycldnn::conv2d::Conv2DParams;
  using Conv2DSizes = sycldnn::conv2d::ConvSizes;

 public:
  // Adds the convolution parameters to the counter set.
  void add_param_counters(State& state, Conv2DParams const& params);

  // Adds theoretical best-case bandwidth requirements to the counter set.
  template <typename T>
  void add_bandwidth_counters(State& state, Conv2DSizes const& sizes);

  // Records the number of elements processed to the counter set. How this
  // calculated varies based on the type of convolution.
  template <typename ConvType>
  void set_items_processed(State& state, Conv2DParams const& params);
};

// Add a full set of counters corresponding to the convolution parameters.
void BaseConvolutionBenchmark::add_param_counters(
    benchmark::State& state, sycldnn::conv2d::Conv2DParams const& params) {
  state.counters["batch"] = params.batch;
  state.counters["in_rows"] = params.in_rows;
  state.counters["in_cols"] = params.in_cols;
  state.counters["channels"] = params.channels;
  state.counters["out_rows"] = params.out_rows;
  state.counters["out_cols"] = params.out_cols;
  state.counters["features"] = params.features;
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
void BaseConvolutionBenchmark::add_bandwidth_counters(
    benchmark::State& state, sycldnn::conv2d::ConvSizes const& sizes) {
  // Compute the size of each element in bytes.
  auto element_bytes = sizeof(ElementType);

  state.counters["bytes_read"] =
      (sizes.filter_size + sizes.input_size) * element_bytes;
  state.counters["bytes_written"] = sizes.output_size * element_bytes;
}

// Records the number of elements processed to the counter set. How this
// calculated varies based on the type of convolution.
template <>
inline void BaseConvolutionBenchmark::set_items_processed<
    sycldnn::conv2d::conv_type::Forward>(
    benchmark::State& state, sycldnn::conv2d::Conv2DParams const& params) {
  state.SetItemsProcessed(state.iterations() * params.batch * params.out_rows *
                          params.out_cols * params.window_rows *
                          params.window_cols * params.channels *
                          params.features * 2);
}

template <>
inline void BaseConvolutionBenchmark::set_items_processed<
    sycldnn::conv2d::conv_type::InputBackprop>(
    benchmark::State& state, sycldnn::conv2d::Conv2DParams const& params) {
  state.SetItemsProcessed(state.iterations() * params.batch * params.in_rows *
                          params.in_cols * params.window_rows *
                          params.window_cols * params.channels *
                          params.features * 2);
}

template <>
inline void BaseConvolutionBenchmark::set_items_processed<
    sycldnn::conv2d::conv_type::FilterBackprop>(
    benchmark::State& state, sycldnn::conv2d::Conv2DParams const& params) {
  state.SetItemsProcessed(state.iterations() * params.batch * params.in_rows *
                          params.in_cols * params.window_rows *
                          params.window_cols * params.channels *
                          params.features * 2);
}

#endif  // define PORTDNN_BENCH_FIXTURE_H_
