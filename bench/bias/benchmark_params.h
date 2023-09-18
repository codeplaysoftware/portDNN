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
#ifndef PORTDNN_BENCH_BIAS_BENCHMARK_PARAMS_H_
#define PORTDNN_BENCH_BIAS_BENCHMARK_PARAMS_H_

#include "portdnn/binaryop/params.h"

#include <benchmark/benchmark.h>

#include <vector>

/**
 * Namespace containing bias parameter serialization and deserialization
 * routines to allow them to be passed into benchmarks at runtime.
 */
namespace benchmark_params {

/**
 * Encode bias parameters as a vector.
 *
 * By passing this vector as an argument to a benchmark::internal::Benchmark
 * instance, these parameters can be provided to each benchmark::State for that
 * benchmark.
 */
inline std::vector<int> serialize(int batch, int rows, int cols, int channels,
                                  int biases) {
  return {batch * rows * cols * channels, biases};
}

/**
 * Extract bias parameters from a benchmark::State instance.
 *
 * Expects the parameters of the benchmark::State to match those provided by the
 * serialize function.
 */
inline sycldnn::binaryop::BinaryParams deserialize(
    benchmark::State const& state) {
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {state.range(0) / state.range(1), state.range(1)};
  params.rhs_dims = {1, state.range(1)};
  return params;
}

}  // namespace benchmark_params

#endif  // PORTDNN_BENCH_BIAS_BENCHMARK_PARAMS_H_
