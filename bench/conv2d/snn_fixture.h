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
#ifndef SYCLDNN_BENCH_CONV2D_SNN_FIXTURE_H_
#define SYCLDNN_BENCH_CONV2D_SNN_FIXTURE_H_

#include "base_convolution_fixture.h"
#include "snn_conv2d_executor.h"

#include "bench/fixture/add_sycl_device_info.h"
#include "bench/fixture/backend_provider.h"
#include "bench/fixture/statistic.h"
#include "bench/fixture/string_reporter.h"
#include "bench/fixture/typenames.h"

template <typename Backend, typename ParamGen, typename ConvType,
          typename Selector>
class SNNConvolutionBenchmark
    : public sycldnn::bench::SNNConv2DExecutor<
          SNNConvolutionBenchmark<Backend, ParamGen, ConvType, Selector>,
          ConvType>,
      public sycldnn::bench::BackendProvider<Backend>,
      public sycldnn::bench::StringReporter,
      public BaseConvolutionBenchmark {
 private:
  using State = benchmark::State;

 protected:
  void run(State& state) {
    auto params = ParamGen()();
    auto selector = Selector();
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::MaxStatistic{}});
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::MinStatistic{}});
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::StdDevStatistic{}});
    this->execute(state, params, selector);

    // Get the SYCL device, and add device and driver info to the benchmark.
    auto backend = this->get_backend();
    auto dev = backend.get_queue().get_device();
    sycldnn::bench::device_info::add_opencl_device_info(dev, *this);

    this->add_to_label("@conv_type", sycldnn::bench::TypeName<ConvType>::name);
    this->add_to_label("@selector", selector.name());
    this->add_to_label("@backend", backend.name());
    this->add_to_label("short_name", "Convolution");
    this->add_to_label("git_hash", commit_hash);
    this->set_label(state);
  };

  void set_model(const char* model_name) {
    this->add_to_label("@model_name", model_name);
  }
};

#define CONVOLUTION_BENCHMARK(model, name, ...)                           \
  BENCHMARK_TEMPLATE_DEFINE_F(SNNConvolutionBenchmark, name, __VA_ARGS__) \
  (benchmark::State & state) {                                            \
    this->set_model(model);                                               \
    this->run(state);                                                     \
  }                                                                       \
  BENCHMARK_REGISTER_F(SNNConvolutionBenchmark, name)                     \
      ->UseManualTime()                                                   \
      ->Unit(benchmark::kNanosecond);

#endif  // SYCLDNN_BENCH_CONV2D_SNN_FIXTURE_H_
