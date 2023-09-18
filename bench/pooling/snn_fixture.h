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
#ifndef PORTDNN_BENCH_POOLING_SNN_FIXTURE_H_
#define PORTDNN_BENCH_POOLING_SNN_FIXTURE_H_

#include "base_pooling_fixture.h"
#include "benchmark_config.h"
#include "benchmark_params.h"
#include "snn_pooling_executor.h"

#include "src/backend/backend_provider.h"

#include "bench/fixture/add_computecpp_info.h"
#include "bench/fixture/add_datatype_info.h"
#include "bench/fixture/add_sycl_device_info.h"
#include "bench/fixture/operator_typenames.h"
#include "bench/fixture/statistic.h"
#include "bench/fixture/string_reporter.h"
#include "bench/fixture/typenames.h"

template <typename Backend, typename DataType, typename Direction,
          template <typename> class Operator>
class SNNPoolingBenchmark
    : public sycldnn::bench::SNNPoolingExecutor<
          SNNPoolingBenchmark<Backend, DataType, Direction, Operator>,
          Direction, Operator>,
      public sycldnn::backend::BackendProvider<Backend>,
      public sycldnn::bench::StringReporter,
      public BasePoolingBenchmark {
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

    // Get the SYCL device, and add device and driver info to the benchmark.
    auto& backend = this->get_backend();
    auto dev = backend.get_queue().get_device();
    sycldnn::bench::device_info::add_opencl_device_info(dev, *this);
    sycldnn::bench::computecpp_info::add_computecpp_version(*this);
    sycldnn::bench::datatype_info::add_datatype_info<DataType>(*this);

    this->add_to_label("@operator",
                       sycldnn::bench::OperatorTypeName<Operator>::name);
    this->add_to_label("@direction", sycldnn::bench::TypeName<Direction>::name);
    this->add_to_label("@library", "portDNN");
    this->add_to_label("@backend", backend.name());
    this->add_to_label("short_name", "Pooling");
    this->add_to_label("git_hash", commit_hash);
    this->set_label(state);
  }

  void set_model(const char* model_name) {
    this->add_to_label("@model_name", model_name);
  }
};

#define POOLING_BENCHMARK(name, ...)                                  \
  BENCHMARK_TEMPLATE_DEFINE_F(SNNPoolingBenchmark, name, __VA_ARGS__) \
  (benchmark::State & state) {                                        \
    this->set_model(get_benchmark_name());                            \
    this->run(state);                                                 \
  }                                                                   \
  BENCHMARK_REGISTER_F(SNNPoolingBenchmark, name)                     \
      ->UseManualTime()                                               \
      ->Unit(benchmark::kNanosecond)                                  \
      ->Apply(RunForAllParamSets);

#endif  // define PORTDNN_BENCH_POOLING_SNN_FIXTURE_H_
