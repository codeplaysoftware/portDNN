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
#ifndef SYCLDNN_BENCH_REDUCTION_BENCHMARK_PARAMS_H_
#define SYCLDNN_BENCH_REDUCTION_BENCHMARK_PARAMS_H_

#include <array>
#include <vector>

#include <benchmark/benchmark.h>

/**
 * Namespace containing reduction parameter serialization and deserialization
 * routines to allow them to be passed into benchmarks at runtime.
 */
namespace benchmark_params {

/**
 * Encode reduction parameters as a vector.
 *
 * By passing this vector as an argument to a benchmark::internal::Benchmark
 * instance, these parameters can be provided to each benchmark::State for that
 * benchmark.
 */
inline std::vector<int> serialize(std::vector<int> dims,
                                  std::vector<int> axis) {
  return {dims[0], dims[1], dims[2], dims[3], axis[0], axis[1]};
}

/**
 * Extract reduction parameters from a benchmark::State instance.
 *
 * Expects the parameters of the benchmark::State to match those provided by the
 * serialize function.
 */
inline std::array<std::vector<int>, 2> deserialize(
    benchmark::State const& state) {
  return std::array<std::vector<int>, 2>{
      std::vector<int>{state.range(0), state.range(1), state.range(2),
                       state.range(3)},
      std::vector<int>{state.range(4), state.range(5)}};
}

}  // namespace benchmark_params

#endif  // SYCLDNN_BENCH_REDUCTION_BENCHMARK_PARAMS_H_
