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
#ifndef PORTDNN_BENCH_BIAS_SNN_BIAS_EXECUTOR_H_
#define PORTDNN_BENCH_BIAS_SNN_BIAS_EXECUTOR_H_

#include <benchmark/benchmark.h>

#include "portdnn/helpers/dims.h"
#include "portdnn/helpers/handle_exception.h"
#include "portdnn/helpers/scope_exit.h"

#include "portdnn/binaryop/launch.h"
#include "portdnn/binaryop/operators.h"
#include "portdnn/binaryop/params.h"

#include "bench/fixture/base_executor.h"

namespace sycldnn {
namespace bench {

/** Executor to perform the bias benchmark using portDNN.  */
template <typename Benchmark>
struct SNNBiasExecutor : public BaseExecutor {
 private:
  using State = ::benchmark::State;
  using BinaryParams = binaryop::BinaryParams;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

 public:
  /** Execute the bias benchmark for the given parameters. */
  void execute(State& state, BinaryParams const& params) {
    auto& benchmark = underlying_benchmark();
    auto& backend = benchmark.get_backend();

    auto lhs_size = sycldnn::helpers::get_total_size(params.lhs_dims);
    auto rhs_size = sycldnn::helpers::get_total_size(params.rhs_dims);
    std::vector<float> inp_vec(lhs_size);
    std::vector<float> bias_vec(rhs_size);
    std::vector<float> out_vec(lhs_size);

    auto inp_gpu =
        benchmark.get_initialised_device_memory(inp_vec.size(), inp_vec);
    auto bias_gpu =
        benchmark.get_initialised_device_memory(bias_vec.size(), bias_vec);
    auto out_gpu =
        benchmark.get_initialised_device_memory(out_vec.size(), out_vec);

    SNN_ON_SCOPE_EXIT {
      benchmark.deallocate_ptr(out_gpu);
      benchmark.deallocate_ptr(bias_gpu);
      benchmark.deallocate_ptr(inp_gpu);
    };

    {  // Ensure the kernel is built before benchmarking
      SNNStatus status;
      try {
        status = sycldnn::binaryop::launch<float, sycldnn::binaryop::Add>(
            inp_gpu, bias_gpu, out_gpu, params, backend);
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
        auto status = sycldnn::binaryop::launch<float, sycldnn::binaryop::Add>(
            inp_gpu, bias_gpu, out_gpu, params, backend);

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

#endif  // PORTDNN_BENCH_BIAS_SNN_BIAS_EXECUTOR_H_
