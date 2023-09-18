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
#ifndef PORTDNN_BENCH_CONV2D_ARM_FIXTURE_H_
#define PORTDNN_BENCH_CONV2D_ARM_FIXTURE_H_

#include "arm_depthwise_conv2d_executor.h"
#include "base_depthwise_convolution_fixture.h"
#include "benchmark_config.h"
#include "benchmark_params.h"

#include "bench/fixture/add_datatype_info.h"

#include "bench/fixture/statistic.h"
#include "bench/fixture/string_reporter.h"
#include "bench/fixture/typenames.h"

extern const char* commit_date;
extern const char* commit_hash;

template <typename DataType, typename ACLExecutor>
class ARMDepthwiseConvolutionBenchmark
    : public sycldnn::bench::ARMDepthwiseConv2DExecutor<
          ARMDepthwiseConvolutionBenchmark<DataType, ACLExecutor>, ACLExecutor>,
      public sycldnn::bench::StringReporter,
      public BaseDepthwiseConvolutionBenchmark {
 private:
  using State = benchmark::State;

 protected:
  void run(State& state) {
    auto params = benchmark_params::deserialize(state);
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::MaxStatistic{}});
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::MinStatistic{}});
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::StdDevStatistic{}});
    this->execute(state, params);

    sycldnn::bench::datatype_info::add_datatype_info<DataType>(*this);

    this->add_to_label("@conv_type", "Forward");
    this->add_to_label("@selector", "ARMCompute");
    this->add_to_label("@library", "ARMComputeLibrary");
    this->add_to_label("short_name", "Depthwise Convolution");
    this->add_to_label("git_hash", commit_hash);
    this->set_label(state);
  };

  void set_model(const char* model_name) {
    this->add_to_label("@model_name", model_name);
  }
};

#define DEPTHWISE_CONVOLUTION_BENCHMARK(name, ...)                    \
  BENCHMARK_TEMPLATE_DEFINE_F(ARMDepthwiseConvolutionBenchmark, name, \
                              __VA_ARGS__)                            \
  (benchmark::State & state) {                                        \
    this->set_model(get_benchmark_name());                            \
    this->run(state);                                                 \
  }                                                                   \
  BENCHMARK_REGISTER_F(ARMDepthwiseConvolutionBenchmark, name)        \
      ->UseManualTime()                                               \
      ->Unit(benchmark::kNanosecond)                                  \
      ->Apply(RunForAllParamSets);

#endif  // define PORTDNN_BENCH_CONV2D_ARM_FIXTURE_H_
