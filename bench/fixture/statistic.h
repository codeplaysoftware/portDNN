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
#ifndef PORTDNN_BENCH_FIXTURE_STATISTIC_H_
#define PORTDNN_BENCH_FIXTURE_STATISTIC_H_

#include <benchmark/benchmark.h>

#include <chrono>
#include <cmath>
#include <limits>

namespace sycldnn {
namespace bench {
/**
 * Abstract Statistic to report for a benchmark.
 */
struct Statistic {
  using Seconds = std::chrono::duration<double>;
  /**
   * Add a time of a single iteration to the Statistic.
   *
   * This should be called once per benchmark iteration.
   *
   * \param sample_time The time of a single benchmark iteration in seconds
   */
  virtual void add_iteration_time(Seconds sample_time) = 0;

  /**
   * Append the Statistic's result to the benchmark state.
   *
   * This should be called after the benchmark loop to output the Statistic's
   * result.
   *
   * \param state The benchmark state to output the results through.
   */
  virtual void add_result_to(::benchmark::State& state) = 0;

  /** Virtual destructor to ensure the implementation's destructor is called. */
  virtual ~Statistic(){};
};

/**
 * Statistic to report the maximum iteration time.
 */
struct MaxStatistic final : Statistic {
  using Seconds = std::chrono::duration<double>;
  using NanoSeconds = std::chrono::duration<double, std::nano>;

  void add_iteration_time(Seconds sample) override {
    if (sample > max_time_) {
      max_time_ = sample;
    }
  }

  void add_result_to(::benchmark::State& state) override {
    auto max_ns = std::chrono::duration_cast<NanoSeconds>(max_time_);
    state.counters["max_time_ns"] = max_ns.count();
  }

 private:
  Seconds max_time_{std::numeric_limits<double>::lowest()};
  ;
};

/**
 * Statistic to report the minimum iteration time.
 */
struct MinStatistic final : Statistic {
  using Seconds = std::chrono::duration<double>;
  using NanoSeconds = std::chrono::duration<double, std::nano>;

  void add_iteration_time(Seconds sample) override {
    if (sample < min_time_) {
      min_time_ = sample;
    }
  }

  void add_result_to(::benchmark::State& state) override {
    auto min_ns = std::chrono::duration_cast<NanoSeconds>(min_time_);
    state.counters["min_time_ns"] = min_ns.count();
  }

 private:
  Seconds min_time_{std::numeric_limits<double>::max()};
};

/**
 * Statistic to use a running total to compute the mean and standard deviation.
 */
struct StdDevStatistic final : Statistic {
  using Seconds = std::chrono::duration<double>;
  using NanoSeconds = std::chrono::duration<double, std::nano>;

  void add_iteration_time(Seconds seconds) override {
    n_samples_++;
    if (n_samples_ == 1) {
      // This is the first sample, so set the mean and leave the variance as 0
      mean_ = seconds;
    } else {
      auto previous_mean = mean_;
      mean_ += (seconds - previous_mean) / n_samples_;
      variance_multiple_ +=
          (seconds - previous_mean).count() * (seconds - mean_).count();
    }
  }

  void add_result_to(::benchmark::State& state) override {
    auto variance = n_samples_ > 1 ? variance_multiple_ / (n_samples_ - 1) : 0.;
    auto std_dev = Seconds{std::sqrt(variance)};

    auto mean_ns = std::chrono::duration_cast<NanoSeconds>(mean_);
    auto std_dev_ns = std::chrono::duration_cast<NanoSeconds>(std_dev);
    state.counters["mean_ns"] = mean_ns.count();
    state.counters["std_dev_ns"] = std_dev_ns.count();
  }

 private:
  int n_samples_{0};
  Seconds mean_{0.};
  // Variance has a unit of essentially seconds * seconds, so cannot store it as
  // a chrono::duration.
  double variance_multiple_{0.};
};

}  // namespace bench
}  // namespace sycldnn

#endif  // PORTDNN_BENCH_FIXTURE_STATISTIC_H_
