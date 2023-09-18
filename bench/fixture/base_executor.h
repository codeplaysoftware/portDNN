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
#ifndef PORTDNN_BENCH_FIXTURE_BASE_EXECUTOR_H_
#define PORTDNN_BENCH_FIXTURE_BASE_EXECUTOR_H_

#include <benchmark/benchmark.h>

#include "bench/fixture/statistic.h"

#include <chrono>
#include <memory>
#include <vector>

namespace sycldnn {
namespace bench {

struct BaseExecutor {
  using Seconds = std::chrono::duration<double>;
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = typename Clock::time_point;

  /**
   * Add a Statistic to be reported by this benchmark.
   *
   * The benchmark will take ownership of the pointer.
   */
  void add_statistic(std::unique_ptr<Statistic>&& stat) {
    statistics_.push_back(std::move(stat));
  }

  /**
   * Start timing the benchmark iteration.
   *
   * The duration between this and the following \ref end_timing() call is used
   * as the iteration time when \ref set_iteration_time() is called.
   *
   * Currently we do not support calling this multiple times in one iteration.
   */
  void start_timing() { start_ = Clock::now(); }

  /**
   * End timing the benchmark iteration.
   *
   * The duration between the previous \ref start_timing() call and this is used
   * as the iteration time when \ref set_iteration_time() is called.
   *
   * Currently we do not support calling this multiple times in one iteration.
   */
  void end_timing() { end_ = Clock::now(); }

  /**
   * Use the benchmark timing information to set the iteration time.
   *
   * Will pass the iteration time to any attached \ref Statistic objects and the
   * benchmark::State.
   */
  void set_iteration_time(::benchmark::State& state) {
    auto elapsed_seconds = std::chrono::duration_cast<Seconds>(end_ - start_);
    state.SetIterationTime(elapsed_seconds.count());
    for (auto& statistic : statistics_) {
      statistic->add_iteration_time(elapsed_seconds);
    }
  }

  /**
   * Add any attached \ref Statistic object's outputs to the benchmark state.
   *
   * Expects to be called once at the end of the benchmark.
   */
  void finish_benchmark(::benchmark::State& state) {
    for (auto& statistic : statistics_) {
      statistic->add_result_to(state);
    }
  }

 protected:
  static constexpr auto AllocationFailure =
      "Error in allocating workspace buffer. The buffer size is likely to be "
      "larger than available device memory.";
  static constexpr auto UnsupportedFailure =
      "Invalid or unsupported benchmark configuration. This may be expected "
      "behaviour and does not indicate a problem.";
  static constexpr auto UnexpectedFailure =
      "This is definitely not expected behaviour and indicates a problem.";

 private:
  std::vector<std::unique_ptr<Statistic>> statistics_;

  TimePoint start_;
  TimePoint end_;
};

}  // namespace bench
}  // namespace sycldnn

#endif  // PORTDNN_BENCH_FIXTURE_BASE_EXECUTOR_H_
