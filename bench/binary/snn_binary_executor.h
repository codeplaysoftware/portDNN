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
#ifndef SYCLDNN_BENCH_BINARY_SNN_BINARY_EXECUTOR_H_
#define SYCLDNN_BENCH_BINARY_SNN_BINARY_EXECUTOR_H_

#include <benchmark/benchmark.h>

#include "sycldnn/helpers/dims.h"
#include "sycldnn/helpers/handle_exception.h"
#include "sycldnn/helpers/scope_exit.h"

#include "sycldnn/binaryop/launch.h"
#include "sycldnn/binaryop/operators.h"
#include "sycldnn/binaryop/params.h"

#include "bench/fixture/base_bench_executor.h"

namespace sycldnn {
namespace bench {

/** Executor to perform the binary benchmark using SYCL-DNN.  */
template <typename Benchmark, typename Operator>
struct SNNBinaryExecutor;

template <typename Benchmark, typename Operator>
struct SNNBinaryExecutor : public BaseBenchExecutor {
 private:
  using State = ::benchmark::State;
  using BinaryParams = binaryop::BinaryParams;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

 public:
  /** Execute the binary benchmark for the given parameters. */
  void execute(State& state, BinaryParams const& params) {
    auto& benchmark = underlying_benchmark();
    auto& backend = benchmark.get_backend();

    auto lhs_size = sycldnn::helpers::get_total_size(params.lhs_dims);
    auto rhs_size = sycldnn::helpers::get_total_size(params.rhs_dims);
    std::vector<float> lhs_vec(lhs_size);
    std::vector<float> rhs_vec(rhs_size);
    std::vector<float> out_vec(lhs_size);

    auto lhs_gpu =
        benchmark.get_initialised_device_memory(lhs_vec.size(), lhs_vec);
    auto rhs_gpu =
        benchmark.get_initialised_device_memory(rhs_vec.size(), rhs_vec);
    auto out_gpu =
        benchmark.get_initialised_device_memory(out_vec.size(), out_vec);

    SNN_ON_SCOPE_EXIT {
      benchmark.deallocate_ptr(out_gpu);
      benchmark.deallocate_ptr(rhs_gpu);
      benchmark.deallocate_ptr(lhs_gpu);
    };

    {  // Ensure the kernel is built before benchmarking
      SNNStatus status;
      try {
        status = sycldnn::binaryop::launch<float, Operator>(
            lhs_gpu, rhs_gpu, out_gpu, params, backend);
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
        auto status = sycldnn::binaryop::launch<float, Operator>(
            lhs_gpu, rhs_gpu, out_gpu, params, backend);

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

#endif  // SYCLDNN_BENCH_BINARY_SNN_BINARY_EXECUTOR_H_
