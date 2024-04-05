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
#ifndef SYCLDNN_BENCH_CONCAT_SNN_FIXTURE_H_
#define SYCLDNN_BENCH_CONCAT_SNN_FIXTURE_H_

#include "base_concat_fixture.h"
#include "snn_concat_executor.h"

#include "src/backend/backend_provider.h"

#include "bench/fixture/add_computecpp_info.h"
#include "bench/fixture/add_datatype_info.h"
#include "bench/fixture/add_sycl_device_info.h"
#include "bench/fixture/operator_typenames.h"
#include "bench/fixture/statistic.h"
#include "bench/fixture/string_reporter.h"
#include "bench/fixture/typenames.h"

template <typename Backend, typename DataType>
class SNNConcatBenchmark : public sycldnn::bench::SNNConcatExecutor<
                               Backend, SNNConcatBenchmark<Backend, DataType>>,
                           public sycldnn::backend::BackendProvider<Backend>,
                           public sycldnn::bench::StringReporter,
                           public BaseConcatBenchmark {
 private:
  using State = benchmark::State;

 protected:
  void run(State& state, std::vector<std::vector<int>>& tensor_descs, int rank,
           int axis) {
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::MaxStatistic{}});
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::MinStatistic{}});
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::StdDevStatistic{}});

    this->execute(state, tensor_descs, rank, axis);

    // Get the SYCL device, and add device and driver info to the benchmark
    auto& backend = this->get_backend();
    auto dev = backend.get_queue().get_device();
    sycldnn::bench::device_info::add_opencl_device_info(dev, *this);
    sycldnn::bench::computecpp_info::add_computecpp_version(*this);
    sycldnn::bench::datatype_info::add_datatype_info<DataType>(*this);

    this->add_to_label("@rank", std::to_string(rank));
    this->add_to_label("@library", "SYCL-DNN");
    this->add_to_label("@backend", backend.name());
    this->add_to_label("short_name", "Concat");
    this->add_to_label("git_hash", commit_hash);
    this->set_label(state);
  }
};

#define CONCAT_BENCHMARK(name, tensor_descs, rank, axis, ...)                \
  BENCHMARK_TEMPLATE_DEFINE_F(SNNConcatBenchmark, name, __VA_ARGS__)         \
  (benchmark::State & state) { this->run(state, tensor_descs, rank, axis); } \
  BENCHMARK_REGISTER_F(SNNConcatBenchmark, name)                             \
      ->UseManualTime()                                                      \
      ->Unit(benchmark::kNanosecond);

#endif