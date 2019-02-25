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
#ifndef SYCLDNN_BENCH_POINTWISE_SNN_POINTWISE_EXECUTOR_H_
#define SYCLDNN_BENCH_POINTWISE_SNN_POINTWISE_EXECUTOR_H_

#include <benchmark/benchmark.h>

#include "sycldnn/helpers/scope_exit.h"

#include "sycldnn/pointwise/direction.h"
#include "sycldnn/pointwise/launch.h"
#include "sycldnn/pointwise/operators.h"

#include "bench/fixture/base_executor.h"

namespace sycldnn {
namespace bench {

/** Executor to perform the pointwise benchmark using SYCL-DNN.  */
template <typename Benchmark, typename Direction,
          template <typename> class Operator>
struct SNNPointwiseExecutor;

template <typename Benchmark, template <typename> class Operator>
struct SNNPointwiseExecutor<Benchmark, sycldnn::pointwise::Forward, Operator>
    : public BaseExecutor {
 private:
  using Forward = sycldnn::pointwise::Forward;
  using State = ::benchmark::State;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

 public:
  /** Execute the pointwise benchmark for the given parameters. */
  void execute(State& state, size_t const n_items) {
    auto& benchmark = underlying_benchmark();
    auto& backend = benchmark.get_backend();

    std::vector<float> inp_vec(n_items);
    std::vector<float> out_vec(n_items);

    auto inp_gpu =
        benchmark.get_initialised_device_memory(inp_vec.size(), inp_vec);
    auto out_gpu =
        benchmark.get_initialised_device_memory(out_vec.size(), out_vec);

    SNN_ON_SCOPE_EXIT {
      benchmark.deallocate_ptr(out_gpu);
      benchmark.deallocate_ptr(inp_gpu);
    };

    {  // Ensure the kernel is built before benchmarking
      auto status = sycldnn::pointwise::launch<float, Operator, Forward>(
          inp_gpu, out_gpu, n_items, backend);

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
      auto status = sycldnn::pointwise::launch<float, Operator, Forward>(
          inp_gpu, out_gpu, n_items, backend);

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

    benchmark.template set_bytes_processed<float>(state, n_items);
    benchmark.add_param_counters(state, n_items);
    benchmark.template add_bandwidth_counters<float, Forward>(state, n_items);

    this->finish_benchmark(state);
  }
};

/**
 * Specialized executor to perform the backprop pointwise benchmark using
 * SYCL-DNN.
 *
 * Pointwise gradients requires the output buffer from the forward pass as
 * well as the backprop buffers. This requires a different executor which
 * can provide the extra buffer.
 */
template <typename Benchmark, template <typename> class Operator>
struct SNNPointwiseExecutor<Benchmark, sycldnn::pointwise::Gradient, Operator>
    : public BaseExecutor {
 private:
  using Gradient = sycldnn::pointwise::Gradient;
  using State = ::benchmark::State;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

 public:
  /** Execute the pointwise benchmark for the given parameters. */
  void execute(State& state, size_t n_items) {
    auto& benchmark = underlying_benchmark();
    auto& backend = benchmark.get_backend();

    std::vector<float> inp_vec(n_items);
    std::vector<float> out_vec(n_items);
    std::vector<float> out_back_vec(n_items);

    auto inp_gpu =
        benchmark.get_initialised_device_memory(inp_vec.size(), inp_vec);
    auto out_gpu =
        benchmark.get_initialised_device_memory(out_vec.size(), out_vec);
    auto out_back_gpu = benchmark.get_initialised_device_memory(
        out_back_vec.size(), out_back_vec);

    {  // Ensure the kernel is built before benchmarking
      auto status = sycldnn::pointwise::launch<float, Operator, Gradient>(
          inp_gpu, out_gpu, out_back_gpu, n_items, backend);
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
      auto status = sycldnn::pointwise::launch<float, Operator, Gradient>(
          inp_gpu, out_gpu, out_back_gpu, n_items, backend);
      status.event.wait_and_throw();
      this->end_timing();
      this->set_iteration_time(state);
    }

    benchmark.deallocate_ptr(out_back_gpu);
    benchmark.deallocate_ptr(out_gpu);
    benchmark.deallocate_ptr(inp_gpu);

    benchmark.template set_bytes_processed<float>(state, n_items);
    benchmark.add_param_counters(state, n_items);
    benchmark.template add_bandwidth_counters<float, Gradient>(state, n_items);

    this->finish_benchmark(state);
  }
};

}  // namespace bench
}  // namespace sycldnn

#endif  // SYCLDNN_BENCH_POINTWISE_SNN_POINTWISE_EXECUTOR_H_
