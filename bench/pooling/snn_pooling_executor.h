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

#include "sycldnn/pooling/launch.h"
#include "sycldnn/pooling/params.h"
#include "sycldnn/pooling/sizes.h"

namespace sycldnn {
namespace bench {

/** Executor to perform the pooling benchmark using SYCL-DNN.  */
template <typename Benchmark, typename Direction,
          template <typename> class Operator>
struct SNNPoolingExecutor {
 private:
  using State = ::benchmark::State;
  using PoolingParams = pooling::PoolingParams;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

 public:
  /** Execute the pooling benchmark for the given parameters. */
  void execute(State& state, PoolingParams const& params) {
    auto& benchmark = underlying_benchmark();
    auto backend = benchmark.get_backend();

    auto pool_sizes = sycldnn::pooling::get_sizes<Direction>(params);

    auto inp_gpu = benchmark.template allocate<float>(pool_sizes.input_size);
    auto out_gpu = benchmark.template allocate<float>(pool_sizes.output_size);

    {  // Ensure the kernel is built before benchmarking
      auto status = sycldnn::pooling::launch<float, Operator, Direction>(
          inp_gpu, out_gpu, params, backend);
      status.event.wait();

      if (sycldnn::StatusCode::OK != status.status) {
        state.SkipWithError(
            "Invalid or unsupported benchmark configuration. "
            "This may be expected behaviour and does not indicate a problem.");
        return;
      }
    }

    for (auto _ : state) {
      auto start = std::chrono::high_resolution_clock::now();
      auto status = sycldnn::pooling::launch<float, Operator, Direction>(
          inp_gpu, out_gpu, params, backend);

      status.event.wait();
      auto end = std::chrono::high_resolution_clock::now();

      auto elapsed_seconds =
          std::chrono::duration_cast<std::chrono::duration<double>>(end -
                                                                    start);

      state.SetIterationTime(elapsed_seconds.count());
    }

    benchmark.deallocate(out_gpu);
    benchmark.deallocate(inp_gpu);

    benchmark.template set_items_processed<Direction>(state, params);
    benchmark.add_param_counters(state, params);
    benchmark.template add_bandwidth_counters<float>(state, pool_sizes);
  }
};

}  // namespace bench
}  // namespace sycldnn

#endif  // SYCLDNN_BENCH_POOLING_SNN_POOLING_EXECUTOR_H_
