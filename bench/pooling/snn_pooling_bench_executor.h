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
#ifndef SYCLDNN_BENCH_POOLING_SNN_POOLING_EXECUTOR_H_
#define SYCLDNN_BENCH_POOLING_SNN_POOLING_EXECUTOR_H_

#include <benchmark/benchmark.h>

#include "src/backend/snn_backend_provider.h"
#include "src/backend/snn_usm_backend_provider.h"
#include "sycldnn/backend/snn_backend.h"
#include "sycldnn/backend/snn_usm_backend.h"

#include "sycldnn/helpers/handle_exception.h"
#include "sycldnn/helpers/scope_exit.h"

#include "sycldnn/pooling/launch.h"
#include "sycldnn/pooling/params.h"
#include "sycldnn/pooling/sizes.h"

#include "bench/common/verification_utils.h"
#include "bench/fixture/base_bench_executor.h"
#include "bench/pooling/snn_pooling_executor.h"

namespace sycldnn {
namespace bench {

/** BenchExecutor to perform the pooling benchmark using SYCL-DNN.  */
template <typename Benchmark, typename Direction,
          template <typename> class Operator, typename Backend,
          bool InOrderQueue>
struct SNNPoolingBenchExecutor : public BaseBenchExecutor {
 private:
  using State = ::benchmark::State;
  using PoolingParams = pooling::PoolingParams;
  using Executor =
      SNNPoolingExecutor<float, Operator, Direction, Backend, InOrderQueue>;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

 public:
  /** Execute the pooling benchmark for the given parameters. */
  void execute(State& state, PoolingParams const& params) {
    if (std::is_same_v<Direction, pooling::Backpropagate> &&
        params.input_format == sycldnn::DataFormat::NCHW) {
      state.SkipWithError(
          (std::string("Backprop does not support NCHW.") + UnsupportedFailure)
              .c_str());
      return;
    }
    auto& benchmark = underlying_benchmark();

    const float max_val = 2.0;
    const auto pool_sizes = sycldnn::pooling::get_sizes<Direction>(params);
    const auto inp_vec =
        internal::iota_initialised_data(pool_sizes.input_size, max_val);

    Executor snn_executor(inp_vec, params);
    snn_executor.setup();

    SNN_ON_SCOPE_EXIT { snn_executor.teardown(); };

    {  // Ensure the kernel is built before benchmarking
      SNNStatus status;
      try {
        status = snn_executor.execute();
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
        auto status = snn_executor.execute();
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

    benchmark.template set_items_processed<Direction>(state, params);
    benchmark.add_param_counters(state, params);
    benchmark.template add_bandwidth_counters<float>(state, pool_sizes);

    this->finish_benchmark(state);
  }
};

/**
 * Specialized executor to perform the max grad pooling benchmark using
 * SYCL-DNN.
 *
 * Max pool gradients require both the original input buffers and the backprop
 * buffers, whereas the other operations do not require both. This requires a
 * different executor which can provide the additional buffers.
 */
template <typename Benchmark, typename Backend, bool InOrderQueue>
struct SNNPoolingBenchExecutor<Benchmark, sycldnn::pooling::Backpropagate,
                               sycldnn::pooling::Max, Backend, InOrderQueue>
    : public BaseBenchExecutor {
 private:
  using State = ::benchmark::State;
  using PoolingParams = pooling::PoolingParams;
  using Executor =
      SNNPoolingExecutor<float, pooling::Max, pooling::Backpropagate, Backend,
                         InOrderQueue>;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

 public:
  /** Execute the pooling benchmark for the given parameters. */
  void execute(State& state, PoolingParams const& params) {
    if (params.input_format == sycldnn::DataFormat::NCHW) {
      state.SkipWithError(
          (std::string("Backprop does not support NCHW.") + UnsupportedFailure)
              .c_str());
      return;
    }
    auto& benchmark = underlying_benchmark();
    const float max_val = 2.0;
    const auto fwd_sizes =
        sycldnn::pooling::get_sizes<pooling::Forward>(params);
    const auto back_sizes =
        sycldnn::pooling::get_sizes<pooling::Backpropagate>(params);
    const auto inp_vec =
        internal::iota_initialised_data(fwd_sizes.input_size, max_val);
    const auto inp_bac_vec =
        internal::iota_initialised_data(back_sizes.input_size, max_val);

    Executor snn_executor(inp_vec, inp_bac_vec, params);
    snn_executor.setup();

    SNN_ON_SCOPE_EXIT { snn_executor.teardown(); };

    {  // Ensure the kernel is built before benchmarking
      SNNStatus status;
      try {
        status = snn_executor.execute();
      } catch (cl::sycl::exception const& e) {
        helpers::handle_exception(e, [&](std::string& msg) {
          state.SkipWithError((msg + UnexpectedFailure).c_str());
        });
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

      if (sycldnn::StatusCode::OK != status.status) {
        state.SkipWithError(UnsupportedFailure);
        return;
      }
    }

    for (auto _ : state) {
      this->start_timing();
      try {
        auto status = snn_executor.execute();
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

    benchmark.template set_items_processed<sycldnn::pooling::Backpropagate>(
        state, params);
    benchmark.add_param_counters(state, params);
    benchmark.template add_bandwidth_counters<float>(state, back_sizes);

    this->finish_benchmark(state);
  }
};

}  // namespace bench
}  // namespace sycldnn

#endif  // SYCLDNN_BENCH_POOLING_SNN_POOLING_EXECUTOR_H_
