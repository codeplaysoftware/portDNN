/*
 * Copyright Codeplay Software Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
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
#ifndef PORTDNN_BENCH_POOLING_BENCHMARK_CONFIG_H_
#define PORTDNN_BENCH_POOLING_BENCHMARK_CONFIG_H_

#include <benchmark/benchmark.h>

#include <vector>

/**
 * Provide a set of pooling benchmark configurations.
 *
 * Each benchmark configuration is a vector of sizes as produced by
 * benchmark_params::serialize, these parameters will then be used to construct
 * the benchmark State. A set of sycldnn::pooling::PoolingParams can be
 * constructed from this State using benchmark_params::deserialize.
 *
 * The definition of this is provided by the specific benchmark models.
 */
std::vector<std::vector<int>> const& get_benchmark_configs();

/**
 * Get the model name to specify in the benchmark output label.
 *
 * The definition of this is provided by the specific benchmark models.
 */
char const* get_benchmark_name();

namespace {
/**
 * Function object to generate all benchmarks from config list, and pass to the
 * benchmarks as runtime parameters.
 */
auto RunForAllParamSets = [](benchmark::internal::Benchmark* b) {
  for (auto& config : get_benchmark_configs()) {
    b->Args(config);
  }
};
}  // namespace

#endif  // PORTDNN_BENCH_POOLING_BENCHMARK_CONFIG_H_
