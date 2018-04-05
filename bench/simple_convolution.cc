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
#include <benchmark/benchmark.h>

#define EIGEN_USE_SYCL
#include "unsupported/Eigen/CXX11/Tensor"

// Need to ensure that Eigen is included before the backend.
// The backend itself doesn't include Eigen to allow useres of SYCL-DNN to
// include it however they wish.
#include "sycldnn/backend/eigen_backend.h"

#include "sycldnn/conv2d/launch.h"
#include "sycldnn/conv2d/params.h"

#include "sycldnn/conv2d/selector/direct_selector.h"
#include "sycldnn/conv2d/selector/tiled_selector.h"

template <typename ConvType>
static void RunConvolutionBM(benchmark::State& state,
                             sycldnn::conv2d::Conv2DParams const& params,
                             sycldnn::conv2d::Selector& selector) {
  Eigen::QueueInterface queue_interface{cl::sycl::default_selector{}};
  Eigen::SyclDevice device{&queue_interface};
  sycldnn::backend::EigenBackend backend{device};

  auto conv_sizes = sycldnn::conv2d::get_sizes<ConvType>(params);

  size_t inp_bytes = conv_sizes.input_size * sizeof(float);
  float* inp_gpu = static_cast<float*>(device.allocate(inp_bytes));
  size_t fil_bytes = conv_sizes.filter_size * sizeof(float);
  float* fil_gpu = static_cast<float*>(device.allocate(fil_bytes));
  size_t out_bytes = conv_sizes.output_size * sizeof(float);
  float* out_gpu = static_cast<float*>(device.allocate(out_bytes));

  {  // Ensure the kernel is built before benchmarking
    auto status = sycldnn::conv2d::launch<float, ConvType>(
        inp_gpu, fil_gpu, out_gpu, params, selector, backend);
    status.event.wait();
  }

  for (auto _ : state) {
    auto start = std::chrono::high_resolution_clock::now();
    auto status = sycldnn::conv2d::launch<float, ConvType>(
        inp_gpu, fil_gpu, out_gpu, params, selector, backend);

    status.event.wait();
    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    state.SetIterationTime(elapsed_seconds.count());
  }

  device.deallocate_all();

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
static void RunForwardConvolutionBM(benchmark::State& state,
                                    sycldnn::conv2d::Conv2DParams const& params,
                                    sycldnn::conv2d::Selector& selector) {
  RunConvolutionBM<sycldnn::conv2d::conv_type::Forward>(state, params,
                                                        selector);
  state.SetItemsProcessed(state.iterations() * params.batch * params.out_rows *
                          params.out_cols * params.window_rows *
                          params.window_cols * params.channels *
                          params.features * 2);
}
static void RunInputBackpropConvolutionBM(
    benchmark::State& state, sycldnn::conv2d::Conv2DParams const& params,
    sycldnn::conv2d::Selector& selector) {
  RunConvolutionBM<sycldnn::conv2d::conv_type::InputBackprop>(state, params,
                                                              selector);
  state.SetItemsProcessed(state.iterations() * params.batch * params.in_rows *
                          params.in_cols * params.window_rows *
                          params.window_cols * params.channels *
                          params.features * 2);
}
static void RunFilterBackpropConvolutionBM(
    benchmark::State& state, sycldnn::conv2d::Conv2DParams const& params,
    sycldnn::conv2d::Selector& selector) {
  RunConvolutionBM<sycldnn::conv2d::conv_type::FilterBackprop>(state, params,
                                                               selector);
  state.SetItemsProcessed(state.iterations() * params.batch * params.in_rows *
                          params.in_cols * params.window_rows *
                          params.window_cols * params.channels *
                          params.features * 2);
}
static sycldnn::conv2d::Conv2DParams get_3x3_params() {
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

template <typename SelectorType>
static void BM_Convolution3x3(benchmark::State& state) {
  auto params = get_3x3_params();
  SelectorType selector{};
  RunForwardConvolutionBM(state, params, selector);
}
template <typename SelectorType>
static void BM_ConvolutionInputBackprop3x3(benchmark::State& state) {
  auto params = get_3x3_params();
  SelectorType selector{};
  RunInputBackpropConvolutionBM(state, params, selector);
}
template <typename SelectorType>
static void BM_ConvolutionFilterBackprop3x3(benchmark::State& state) {
  auto params = get_3x3_params();
  SelectorType selector{};
  RunFilterBackpropConvolutionBM(state, params, selector);
}
BENCHMARK_TEMPLATE(BM_Convolution3x3, sycldnn::conv2d::DirectSelector)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_ConvolutionInputBackprop3x3,
                   sycldnn::conv2d::DirectSelector)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_ConvolutionFilterBackprop3x3,
                   sycldnn::conv2d::DirectSelector)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_Convolution3x3, sycldnn::conv2d::TiledSelector)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_ConvolutionInputBackprop3x3,
                   sycldnn::conv2d::TiledSelector)
    ->UseManualTime();

static sycldnn::conv2d::Conv2DParams get_3x3stride2_params() {
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
template <typename SelectorType>
static void BM_Convolution3x3Stride2(benchmark::State& state) {
  auto params = get_3x3stride2_params();
  SelectorType selector{};
  RunForwardConvolutionBM(state, params, selector);
}
template <typename SelectorType>
static void BM_ConvolutionInputBackprop3x3Stride2(benchmark::State& state) {
  auto params = get_3x3stride2_params();
  SelectorType selector{};
  RunInputBackpropConvolutionBM(state, params, selector);
}
template <typename SelectorType>
static void BM_ConvolutionFilterBackprop3x3Stride2(benchmark::State& state) {
  auto params = get_3x3stride2_params();
  SelectorType selector{};
  RunFilterBackpropConvolutionBM(state, params, selector);
}
BENCHMARK_TEMPLATE(BM_Convolution3x3Stride2, sycldnn::conv2d::DirectSelector)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_ConvolutionInputBackprop3x3Stride2,
                   sycldnn::conv2d::DirectSelector)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_ConvolutionFilterBackprop3x3Stride2,
                   sycldnn::conv2d::DirectSelector)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_Convolution3x3Stride2, sycldnn::conv2d::TiledSelector)
    ->UseManualTime();
BENCHMARK_TEMPLATE(BM_ConvolutionInputBackprop3x3Stride2,
                   sycldnn::conv2d::TiledSelector)
    ->UseManualTime();
