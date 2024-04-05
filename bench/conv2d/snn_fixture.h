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
#ifndef SYCLDNN_BENCH_CONV2D_SNN_FIXTURE_H_
#define SYCLDNN_BENCH_CONV2D_SNN_FIXTURE_H_

#include "base_convolution_fixture.h"
#include "benchmark_config.h"
#include "benchmark_params.h"
#include "snn_conv2d_bench_executor.h"
#include "src/backend/backend_provider.h"

#include "bench/fixture/add_computecpp_info.h"
#include "bench/fixture/add_datatype_info.h"
#include "bench/fixture/add_sycl_device_info.h"
#include "bench/fixture/statistic.h"
#include "bench/fixture/string_reporter.h"
#include "bench/fixture/typenames.h"

#include "sycldnn/conv2d/selector/winograd_selector.h"

#include <vector>

template <typename Selector, typename DataLayout>
struct use_transpose {
  static constexpr bool value =
      (std::is_same_v<Selector, sycldnn::conv2d::WinogradSelector> ||
       std::is_same_v<Selector, sycldnn::conv2d::WinogradLargeSelector>)&&std::
          is_same_v<DataLayout, sycldnn::layout::NCHW>;
};

template <typename Backend, typename DataType, typename DataLayout,
          typename ConvType, typename Selector, bool UseTensorOp,
          bool InOrderQueue>
class SNNConvolutionBenchmark
    : public sycldnn::bench::SNNConv2DBenchExecutor<
          SNNConvolutionBenchmark<Backend, DataType, DataLayout, ConvType,
                                  Selector, UseTensorOp, InOrderQueue>,
          ConvType, Backend, UseTensorOp, InOrderQueue,
          use_transpose<Selector, DataLayout>::value>,
      public sycldnn::backend::BackendProvider<Backend>,
      public sycldnn::bench::StringReporter,
      public BaseConvolutionBenchmark {
 private:
  using State = benchmark::State;

 protected:
  void run(State& state) {
    auto params = benchmark_params::deserialize(state);
    params.input_format = DataLayout::input_layout;
    params.filter_format = DataLayout::filter_layout;
    auto selector = Selector();
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::MaxStatistic{}});
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::MinStatistic{}});
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::StdDevStatistic{}});
    this->execute(state, params, selector);

    // Get the SYCL device, and add device and driver info to the benchmark.
    auto& backend = this->get_backend();
    auto dev = backend.get_queue().get_device();
    sycldnn::bench::device_info::add_opencl_device_info(dev, *this);
    sycldnn::bench::computecpp_info::add_computecpp_version(*this);
    sycldnn::bench::datatype_info::add_datatype_info<DataType>(*this);

    this->add_to_label("@layout", sycldnn::bench::TypeName<DataLayout>::name);
    this->add_to_label("@conv_type", sycldnn::bench::TypeName<ConvType>::name);
    this->add_to_label("@selector", selector.name());
    this->add_to_label("@library", "SYCL-DNN");
    this->add_to_label("@backend", backend.name());
    this->add_to_label("short_name", "Convolution");
    this->add_to_label("git_hash", commit_hash);
    this->add_to_label("@use_tensor_op", UseTensorOp ? "true" : "false");
    this->add_to_label("@in_order_queue", InOrderQueue ? "true" : "false");
    this->add_to_label(
        "@use_transpose",
        use_transpose<Selector, DataLayout>::value ? "true" : "false");
    this->set_label(state);
  }

  void set_model(const char* model_name) {
    this->add_to_label("@model_name", model_name);
  }
};

#define CONVOLUTION_BENCHMARK(name, ...)                                  \
  BENCHMARK_TEMPLATE_DEFINE_F(SNNConvolutionBenchmark, name, __VA_ARGS__) \
  (benchmark::State & state) {                                            \
    this->set_model(get_benchmark_name());                                \
    this->run(state);                                                     \
  }                                                                       \
  BENCHMARK_REGISTER_F(SNNConvolutionBenchmark, name)                     \
      ->UseManualTime()                                                   \
      ->Unit(benchmark::kNanosecond)                                      \
      ->Apply(RunForAllParamSets);

#endif  // SYCLDNN_BENCH_CONV2D_SNN_FIXTURE_H_
