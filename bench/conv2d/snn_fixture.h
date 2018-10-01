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
#include "bench/fixture/eigen_backend_provider.h"
#include "bench/fixture/string_reporter.h"

template <typename ParamGen, typename ConvType, typename Selector>
class SNNConvolutionBenchmark
    : public sycldnn::bench::SNNConv2DExecutor<
          SNNConvolutionBenchmark<ParamGen, ConvType, Selector>, ConvType>,
      public sycldnn::bench::EigenBackendProvider,
      public sycldnn::bench::StringReporter,
      public BaseConvolutionBenchmark {
 private:
  using State = benchmark::State;

 protected:
  void run(State& state) {
    auto params = ParamGen()();
    auto selector = Selector();
    this->execute(state, params, selector);

    // Get the SYCL device, and add device and driver info to the benchmark.
    auto dev = get_backend().get_queue().get_device();
    sycldnn::bench::device_info::add_opencl_device_info(dev, *this);

    this->add_to_label("selector", selector.name());
    this->add_to_label("git_hash", commit_hash);
    this->set_label(state);
  };
};

#define CONVOLUTION_BENCHMARK(name, ...)                                  \
  BENCHMARK_TEMPLATE_DEFINE_F(SNNConvolutionBenchmark, name, __VA_ARGS__) \
  (benchmark::State & state) { this->run(state); }                        \
  BENCHMARK_REGISTER_F(SNNConvolutionBenchmark, name)                     \
      ->UseManualTime()                                                   \
      ->Unit(benchmark::kNanosecond);

#endif  // SYCLDNN_BENCH_CONV2D_SNN_FIXTURE_H_
