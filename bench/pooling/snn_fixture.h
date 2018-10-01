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
#ifndef SYCLDNN_BENCH_POOLING_SNN_FIXTURE_H_
#define SYCLDNN_BENCH_POOLING_SNN_FIXTURE_H_

#include "base_pooling_fixture.h"
#include "snn_pooling_executor.h"

#include "bench/fixture/add_sycl_device_info.h"
#include "bench/fixture/eigen_backend_provider.h"
#include "bench/fixture/string_reporter.h"

template <typename ParamGen, typename Direction,
          template <typename> class Operator>
class SNNPoolingBenchmark
    : public sycldnn::bench::SNNPoolingExecutor<
          SNNPoolingBenchmark<ParamGen, Direction, Operator>, Direction,
          Operator>,
      public sycldnn::bench::EigenBackendProvider,
      public sycldnn::bench::StringReporter,
      public BasePoolingBenchmark {
 private:
  using State = benchmark::State;

 protected:
  void run(State& state) {
    auto params = ParamGen()();
    this->execute(state, params);

    // Get the SYCL device, and add device and driver info to the benchmark.
    auto dev = get_backend().get_queue().get_device();
    sycldnn::bench::device_info::add_opencl_device_info(dev, *this);

    this->add_to_label("git_hash", commit_hash);
    this->set_label(state);
  };
};

#define POOLING_BENCHMARK(name, ...)                                  \
  BENCHMARK_TEMPLATE_DEFINE_F(SNNPoolingBenchmark, name, __VA_ARGS__) \
  (benchmark::State & state) { this->run(state); }                    \
  BENCHMARK_REGISTER_F(SNNPoolingBenchmark, name)                     \
      ->UseManualTime()                                               \
      ->Unit(benchmark::kNanosecond);

#endif  // define SYCLDNN_BENCH_POOLING_SNN_FIXTURE_H_
