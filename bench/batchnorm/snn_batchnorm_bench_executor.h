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
#ifndef SYCLDNN_BENCH_BATCHNORM_SNN_BATCHNORM_BENCH_EXECUTOR_H_
#define SYCLDNN_BENCH_BATCHNORM_SNN_BATCHNORM_BENCH_EXECUTOR_H_

#include <benchmark/benchmark.h>

#include "sycldnn/helpers/handle_exception.h"
#include "sycldnn/helpers/scope_exit.h"

#include "bench/batchnorm/snn_batchnorm_executor.h"

#include "bench/batchnorm/snn_batchnorm_executor.h"
#include "bench/common/event_handling.h"
#include "bench/common/verification_utils.h"
#include "bench/fixture/base_bench_executor.h"

#include "bench/common/verification_utils.h"

namespace sycldnn {
namespace bench {

/** BenchExecutor to perform the batchnorm benchmark using SYCL-DNN.  */
template <typename Benchmark, typename DataType, typename Backend,
          typename Direction, bool InOrderQueue>
struct SNNBatchnormBenchExecutor : public BaseBenchExecutor {
 private:
  using State = ::benchmark::State;
  using Params = batchnorm::BatchNormParams;
  using Executor =
      SNNBatchnormExecutor<DataType, Direction, Backend, InOrderQueue>;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

 public:
  /** Execute the batchnorm benchmark for the given parameters. */
  void execute(State& state, Params const& params) {
    auto& benchmark = underlying_benchmark();

    auto input_size =
        params.batch * params.rows * params.cols * params.channels;

    const DataType max_val = 2.0;
    const auto input_vec = internal::iota_initialised_data(input_size, max_val);
    const auto beta_vec =
        internal::iota_initialised_data(params.channels, max_val);
    const auto gamma_vec =
        internal::iota_initialised_data(params.channels, max_val);
    std::vector<DataType> mean_vec =
        internal::iota_initialised_data(params.channels, max_val);
    std::vector<DataType> var_vec =
        internal::iota_initialised_data(params.channels, max_val);

    Executor snn_batchnorm(input_vec, beta_vec, gamma_vec, mean_vec, var_vec,
                           params);

    try {
      snn_batchnorm.setup();
    } catch (...) {
      state.SkipWithError(AllocationFailure);
      return;
    }

    SNN_ON_SCOPE_EXIT { snn_batchnorm.teardown(); };

    auto queue = snn_batchnorm.get_queue();
    {  // Ensure the kernel is built before benchmarking
      SNNStatus status;
      try {
        status = snn_batchnorm.execute();
      } catch (cl::sycl::exception const& e) {
        helpers::handle_exception(e, [&](std::string& msg) {
          state.SkipWithError((msg + UnexpectedFailure).c_str());
        });
        return;
      }

      if (sycldnn::StatusCode::OK != status.status) {
        state.SkipWithError(UnsupportedFailure);
        return;
      }

      try {
        status.event.wait_and_throw();
      } catch (cl::sycl::exception const& e) {
        helpers::handle_exception(e, [&](std::string& msg) {
          state.SkipWithError((msg + UnexpectedFailure).c_str());
        });
        return;
      } catch (std::exception const& e) {
        helpers::handle_exception(e, [&](std::string& msg) {
          state.SkipWithError((msg + UnexpectedFailure).c_str());
        });
        return;
      }
    }

    for (auto _ : state) {
      this->start_timing();
      try {
        auto status = snn_batchnorm.execute();
        status.event.wait_and_throw();
      } catch (cl::sycl::exception const& e) {
        helpers::handle_exception(e, [&](std::string& msg) {
          state.SkipWithError((msg + UnexpectedFailure).c_str());
        });
        return;
      } catch (std::exception const& e) {
        helpers::handle_exception(e, [&](std::string& msg) {
          state.SkipWithError((msg + UnexpectedFailure).c_str());
        });
        return;
      }

      this->end_timing();
      this->set_iteration_time(state);
    }

    benchmark.set_items_processed(state, params);
    benchmark.add_param_counters(state, params);
    benchmark.template add_bandwidth_counters<float>(state, params);

    this->finish_benchmark(state);
  }
};

}  // namespace bench
}  // namespace sycldnn

#endif  // SYCLDNN_BENCH_BATCHNORM_SNN_BATCHNORM_BENCH_EXECUTOR_H_
