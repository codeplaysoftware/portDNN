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
#ifndef SYCLDNN_BENCH_BINARY_BASE_BINARY_FIXTURE_H_
#define SYCLDNN_BENCH_BINARY_BASE_BINARY_FIXTURE_H_

#include <benchmark/benchmark.h>

#include "sycldnn/binaryop/params.h"
#include "sycldnn/helpers/dims.h"

extern const char* commit_date;
extern const char* commit_hash;

class BaseBinaryBenchmark : public benchmark::Fixture {
 private:
  using State = benchmark::State;
  using BinaryParams = sycldnn::binaryop::BinaryParams;

 public:
  // Adds the binary parameters to the counter set.
  void add_param_counters(State& state, BinaryParams const& params);

  // Adds theoretical best-case bandwidth requirements to the counter set.
  template <typename T>
  void add_bandwidth_counters(State& state, BinaryParams const& params);

  // Records the number of elements processed to the counter set. How this
  // calculated varies based on the type of operation.
  inline void set_items_processed(State& state, BinaryParams const& params);
};

// Add a full set of counters corresponding to the binary parameters.
void BaseBinaryBenchmark::add_param_counters(benchmark::State& state,
                                             BinaryParams const& params) {
  state.counters["lhs_items"] =
      sycldnn::helpers::get_total_size(params.lhs_dims);
  state.counters["rhs_items"] =
      sycldnn::helpers::get_total_size(params.rhs_dims);

  // Tensor 1 parameters
  state.counters["n_lhs"] = params.lhs_dims[0];
  state.counters["h_lhs"] = params.lhs_dims[1];
  state.counters["w_lhs"] = params.lhs_dims[2];
  state.counters["c_lhs"] = params.lhs_dims[3];

  // Tensor 2 parameters
  state.counters["n_rhs"] = params.rhs_dims[0];
  state.counters["h_rhs"] = params.rhs_dims[1];
  state.counters["w_rhs"] = params.rhs_dims[2];
  state.counters["c_rhs"] = params.rhs_dims[3];
}

// Calculate the optimal bandwidth requirements, and add corresponding counters.
// This assumes each binary element is read exactly once, rather than the actual
// behaviour where multiple threads may re-read the same values.
template <typename ElementType>
void BaseBinaryBenchmark::add_bandwidth_counters(benchmark::State& state,
                                                 BinaryParams const& params) {
  // Compute the size of each element in bytes.
  auto element_bytes = sizeof(ElementType);
  auto lhs_size = sycldnn::helpers::get_total_size(params.lhs_dims);
  auto rhs_size = sycldnn::helpers::get_total_size(params.rhs_dims);

  state.counters["bytes_read"] = (lhs_size + rhs_size) * element_bytes;
  state.counters["bytes_written"] = lhs_size * element_bytes;
}

// Records the number of elements processed to the counter set. How this
// is calculated varies based on the type of operation.
inline void BaseBinaryBenchmark::set_items_processed(
    benchmark::State& state, BinaryParams const& params) {
  // We define items processed as neighbourhood size * output tensor size for
  // binary operations.
  auto tensor_size = sycldnn::helpers::get_total_size(params.lhs_dims);

  state.SetItemsProcessed(state.iterations() * tensor_size);
}

#endif  // SYCLDNN_BENCH_BINARY_BASE_BINARY_FIXTURE_H_
