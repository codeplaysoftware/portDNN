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
#ifndef SYCLDNN_BENCH_REDUCTION_SNN_REDUCTION_EXECUTOR_H_
#define SYCLDNN_BENCH_REDUCTION_SNN_REDUCTION_EXECUTOR_H_

#include <benchmark/benchmark.h>

#include "sycldnn/helpers/handle_exception.h"
#include "sycldnn/helpers/scope_exit.h"

#include "sycldnn/reduce/launch.h"
#include "sycldnn/reduce/operators.h"

#include "bench/fixture/base_bench_executor.h"

namespace sycldnn {
namespace bench {

/** Executor to perform the reduction benchmark using SYCL-DNN.  */
template <typename Benchmark, typename Backend, typename DataType, typename Op>
struct SNNReductionExecutor : public BaseBenchExecutor {
 private:
  using State = ::benchmark::State;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

 public:
  /** Execute the reduction benchmark for the given parameters. */
  void execute(State& state, int batch, int outer, int inner) {
    auto& benchmark = underlying_benchmark();
    auto& backend = benchmark.get_backend();

    auto input_size = batch * outer * inner;
    auto output_size = batch * inner;
    std::vector<float> input_vec(input_size);
    std::vector<float> out_vec(output_size);

    auto input_gpu =
        benchmark.get_initialised_device_memory(input_vec.size(), input_vec);
    auto out_gpu =
        benchmark.get_initialised_device_memory(out_vec.size(), out_vec);

    SNN_ON_SCOPE_EXIT {
      benchmark.deallocate_ptr(input_gpu);
      benchmark.deallocate_ptr(out_gpu);
    };

    {
      SNNStatus status;
      try {
        status = sycldnn::reduce::launch<DataType, Op>(
            input_gpu, out_gpu, batch, outer, inner, backend);
        status.event.wait_and_throw();
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
    }

    for (auto _ : state) {
      this->start_timing();
      try {
        auto status = sycldnn::reduce::launch<DataType, Op>(
            input_gpu, out_gpu, batch, outer, inner, backend);

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

    benchmark.set_items_processed(state, batch, outer, inner);
    benchmark.add_param_counters(state, batch, outer, inner);
    benchmark.template add_bandwidth_counters<float>(state, batch, outer,
                                                     inner);

    this->finish_benchmark(state);
  }
};

}  // namespace bench
}  // namespace sycldnn

#endif  // SYCLDNN_BENCH_REDUCTION_SNN_REDUCTION_EXECUTOR_H_
