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
#ifndef PORTDNN_BENCH_BATCHNORM_BENCHMARK_PARAMS_H_
#define PORTDNN_BENCH_BATCHNORM_BENCHMARK_PARAMS_H_

#include "portdnn/batchnorm/params.h"

#include <benchmark/benchmark.h>

#include <vector>

/**
 * Namespace containing batchnorm parameter serialization and deserialization
 * routines to allow them to be passed into benchmarks at runtime.
 */
namespace benchmark_params {

/**
 * Encode batchnorm parameters as a vector.
 *
 * By passing this vector as an argument to a benchmark::internal::Benchmark
 * instance, these parameters can be provided to each benchmark::State for that
 * benchmark.
 */
inline std::vector<int> serialize(int batch, int rows, int cols, int channels) {
  return {batch, rows, cols, channels};
}

/**
 * Extract batchnorm parameters from a benchmark::State instance.
 *
 * Expects the parameters of the benchmark::State to match those provided by the
 * serialize function.
 */
inline sycldnn::batchnorm::BatchNormParams deserialize(
    benchmark::State const& state) {
  sycldnn::batchnorm::BatchNormParams params;
  params.batch = state.range(0);
  params.rows = state.range(1);
  params.cols = state.range(2);
  params.channels = state.range(3);
  params.is_training = false;
  return params;
}

}  // namespace benchmark_params

#endif  // PORTDNN_BENCH_BATCHNORM_BENCHMARK_PARAMS_H_
