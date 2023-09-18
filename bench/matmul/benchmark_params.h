/*
 * Copyright Codeplay Software Ltd
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
#ifndef PORTDNN_BENCH_MATMUL_BENCHMARK_PARAMS_H_
#define PORTDNN_BENCH_MATMUL_BENCHMARK_PARAMS_H_

#include <benchmark/benchmark.h>

#include <array>

/**
 * Namespace containing matmul parameter serialization and deserialization
 * routines to allow them to be passed into benchmarks at runtime.
 */
namespace matmul_benchmark_params {

/**
 * Encode matmul parameters as a vector.
 *
 * By passing this vector as an argument to a benchmark::internal::Benchmark
 * instance, these parameters can be provided to each benchmark::State for that
 * benchmark.
 */
inline std::vector<int> serialize(int m, int k, int n, int batch,
                                  bool transpose_lhs, bool transpose_rhs) {
  return {m, k, n, batch, transpose_lhs, transpose_rhs};
}

struct MatmulParams {
  int m;
  int k;
  int n;
  int batch;
  bool transpose_lhs;
  bool transpose_rhs;
};

/**
 * Extract matmul parameters from a benchmark::State instance.
 *
 * Expects the parameters of the benchmark::State to match those provided by the
 * serialize function.
 */
inline MatmulParams deserialize(benchmark::State const& state) {
  MatmulParams params;
  params.m = state.range(0);
  params.k = state.range(1);
  params.n = state.range(2);
  params.batch = state.range(3);
  params.transpose_lhs = state.range(4);
  params.transpose_rhs = state.range(5);
  return params;
}

}  // namespace matmul_benchmark_params

#endif  // PORTDNN_BENCH_MATMUL_BENCHMARK_PARAMS_H_
