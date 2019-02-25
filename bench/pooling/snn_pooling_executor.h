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
#ifndef SYCLDNN_BENCH_POOLING_SNN_POOLING_EXECUTOR_H_
#define SYCLDNN_BENCH_POOLING_SNN_POOLING_EXECUTOR_H_

#include <benchmark/benchmark.h>

#include "sycldnn/helpers/scope_exit.h"

#include "sycldnn/pooling/launch.h"
#include "sycldnn/pooling/params.h"
#include "sycldnn/pooling/sizes.h"

#include "bench/fixture/base_executor.h"

namespace sycldnn {
namespace bench {

/** Executor to perform the pooling benchmark using SYCL-DNN.  */
template <typename Benchmark, typename Direction,
          template <typename> class Operator>
struct SNNPoolingExecutor : public BaseExecutor {
 private:
  using State = ::benchmark::State;
  using PoolingParams = pooling::PoolingParams;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

 public:
  /** Execute the pooling benchmark for the given parameters. */
  void execute(State& state, PoolingParams const& params) {
    auto& benchmark = underlying_benchmark();
    auto& backend = benchmark.get_backend();

    auto pool_sizes = sycldnn::pooling::get_sizes<Direction>(params);

    std::vector<float> inp_vec(pool_sizes.input_size);
    std::vector<float> out_vec(pool_sizes.output_size);

    auto inp_gpu =
        benchmark.get_initialised_device_memory(inp_vec.size(), inp_vec);
    auto out_gpu =
        benchmark.get_initialised_device_memory(out_vec.size(), out_vec);

    SNN_ON_SCOPE_EXIT {
      benchmark.deallocate_ptr(out_gpu);
      benchmark.deallocate_ptr(inp_gpu);
    };

    {  // Ensure the kernel is built before benchmarking
      auto status = sycldnn::pooling::launch<float, Operator, Direction>(
          inp_gpu, out_gpu, params, backend);

      if (sycldnn::StatusCode::OK != status.status) {
        state.SkipWithError(
            "Invalid or unsupported benchmark configuration. "
            "This may be expected behaviour and does not indicate a problem.");
        return;
      }

      try {
        status.event.wait_and_throw();
      } catch (cl::sycl::exception const& e) {
        auto error = std::string{"cl::sycl::exception caught: "} + e.what() +
                     ". This is definitely not expected behaviour and "
                     "indicates a problem.";
        state.SkipWithError(error.c_str());
        return;
      }
    }

    for (auto _ : state) {
      this->start_timing();
      auto status = sycldnn::pooling::launch<float, Operator, Direction>(
          inp_gpu, out_gpu, params, backend);

      try {
        status.event.wait_and_throw();
      } catch (cl::sycl::exception const& e) {
        auto error = std::string{"cl::sycl::exception caught: "} + e.what() +
                     "This is definitely not expected behaviour and indicates "
                     "a problem.";
        state.SkipWithError(error.c_str());
        return;
      }

      this->end_timing();

      this->set_iteration_time(state);
    }

    // TODO: This wait shouldn't be required once ComputeCpp resolves SYCLE-213.
    backend.get_queue().wait_and_throw();

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
template <typename Benchmark>
struct SNNPoolingExecutor<Benchmark, sycldnn::pooling::Backpropagate,
                          sycldnn::pooling::Max> : public BaseExecutor {
 private:
  using State = ::benchmark::State;
  using PoolingParams = pooling::PoolingParams;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

 public:
  /** Execute the pooling benchmark for the given parameters. */
  void execute(State& state, PoolingParams const& params) {
    auto& benchmark = underlying_benchmark();
    auto& backend = benchmark.get_backend();

    auto fwd_sizes =
        sycldnn::pooling::get_sizes<sycldnn::pooling::Forward>(params);
    auto back_sizes =
        sycldnn::pooling::get_sizes<sycldnn::pooling::Backpropagate>(params);

    std::vector<float> inp_vec(fwd_sizes.input_size);
    std::vector<float> inp_back_vec(back_sizes.input_size);
    std::vector<float> out_vec(back_sizes.output_size);
    std::vector<float> out_back_vec(back_sizes.output_size);

    auto inp_gpu =
        benchmark.get_initialised_device_memory(inp_vec.size(), inp_vec);
    auto out_gpu =
        benchmark.get_initialised_device_memory(out_vec.size(), out_vec);
    auto inp_back_gpu = benchmark.get_initialised_device_memory(
        inp_back_vec.size(), inp_back_vec);
    auto out_back_gpu = benchmark.get_initialised_device_memory(
        out_back_vec.size(), out_back_vec);

    SNN_ON_SCOPE_EXIT {
      benchmark.deallocate_ptr(out_back_gpu);
      benchmark.deallocate_ptr(inp_back_gpu);
      benchmark.deallocate_ptr(out_gpu);
      benchmark.deallocate_ptr(inp_gpu);
    };

    {  // Ensure the kernel is built before benchmarking
      auto status = sycldnn::pooling::launch<float, sycldnn::pooling::Max,
                                             sycldnn::pooling::Backpropagate>(
          inp_gpu, out_gpu, inp_back_gpu, out_back_gpu, params, backend);
      status.event.wait_and_throw();

      if (sycldnn::StatusCode::OK != status.status) {
        state.SkipWithError(
            "Invalid or unsupported benchmark configuration. "
            "This may be expected behaviour and does not indicate a problem.");
        return;
      }
    }

    for (auto _ : state) {
      this->start_timing();
      auto status = sycldnn::pooling::launch<float, sycldnn::pooling::Max,
                                             sycldnn::pooling::Backpropagate>(
          inp_gpu, out_gpu, inp_back_gpu, out_back_gpu, params, backend);
      status.event.wait_and_throw();
      this->end_timing();
      this->set_iteration_time(state);
    }

    // TODO: This wait shouldn't be required once ComputeCpp resolves SYCLE-213.
    backend.get_queue().wait_and_throw();

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
