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
#ifndef SYCLDNN_BENCH_BINARY_BENCHMARK_PARAMS_H_
#define SYCLDNN_BENCH_BINARY_BENCHMARK_PARAMS_H_

#include "sycldnn/binaryop/params.h"

#include <benchmark/benchmark.h>

#include <vector>

/**
 * Namespace containing binary parameter serialization and deserialization
 * routines to allow them to be passed into benchmarks at runtime.
 */
namespace benchmark_params {

/**
 * Encode binary parameters as a vector.
 *
 * By passing this vector as an argument to a benchmark::internal::Benchmark
 * instance, these parameters can be provided to each benchmark::State for that
 * benchmark.
 */
inline std::vector<int> serialize(int batch1, int rows1, int cols1,
                                  int channels1, int batch2, int rows2,
                                  int cols2, int channels2) {
  return {batch1, rows1, cols1, channels1, batch2, rows2, cols2, channels2};
}

/**
 * Extract binary parameters from a benchmark::State instance.
 *
 * Expects the parameters of the benchmark::State to match those provided by the
 * serialize function.
 */
inline sycldnn::binaryop::BinaryParams deserialize(
    benchmark::State const& state) {
  sycldnn::binaryop::BinaryParams params;
  params.lhs_dims = {state.range(0), state.range(1), state.range(2),
                     state.range(3)};
  params.rhs_dims = {state.range(4), state.range(5), state.range(6),
                     state.range(7)};
  return params;
}

}  // namespace benchmark_params

#endif  // SYCLDNN_BENCH_BINARY_BENCHMARK_PARAMS_H_
