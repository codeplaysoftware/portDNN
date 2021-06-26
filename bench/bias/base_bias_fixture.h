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
#ifndef SYCLDNN_BENCH_BIAS_BASE_BIAS_FIXTURE_H_
#define SYCLDNN_BENCH_BIAS_BASE_BIAS_FIXTURE_H_

#include <benchmark/benchmark.h>

#include "sycldnn/bias/sizes.h"

extern const char* commit_date;
extern const char* commit_hash;

class BaseBiasBenchmark : public benchmark::Fixture {
 private:
  using State = benchmark::State;
  using BiasParams = sycldnn::bias::BiasParams;
  using BiasSizes = sycldnn::bias::BiasSizes;

 public:
  // Adds the bias parameters to the counter set.
  void add_param_counters(State& state, BiasParams const& params);

  // Adds theoretical best-case bandwidth requirements to the counter set.
  template <typename T>
  void add_bandwidth_counters(State& state, BiasSizes const& sizes);

  // Records the number of elements processed to the counter set. How this
  // calculated varies based on the type of operation.
  inline void set_items_processed(State& state, BiasParams const& params);
};

// Add a full set of counters corresponding to the bias parameters.
void BaseBiasBenchmark::add_param_counters(benchmark::State& state,
                                           BiasParams const& params) {
  state.counters["batch"] = params.batch;
  state.counters["in_rows"] = params.in_rows;
  state.counters["in_cols"] = params.in_cols;
  state.counters["channels"] = params.channels;
  state.counters["bias"] = params.bias;
}

// Calculate the optimal bandwidth requirements, and add corresponding counters.
// This assumes each bias element is read exactly once, rather than the actual
// behaviour where multiple threads may re-read the same values.
template <typename ElementType>
void BaseBiasBenchmark::add_bandwidth_counters(benchmark::State& state,
                                               BiasSizes const& sizes) {
  // Compute the size of each element in bytes.
  auto element_bytes = sizeof(ElementType);

  state.counters["bytes_read"] =
      (sizes.input_size + sizes.bias_size) * element_bytes;
  state.counters["bytes_written"] = sizes.output_size * element_bytes;
}

// Records the number of elements processed to the counter set. How this
// is calculated varies based on the type of operation.
inline void BaseBiasBenchmark::set_items_processed(benchmark::State& state,
                                                   BiasParams const& params) {
  // We define items processed as neighbourhood size * output tensor size for
  // bias operations.
  auto tensor_size =
      params.batch * params.in_rows * params.in_cols * params.channels;

  state.SetItemsProcessed(state.iterations() * tensor_size);
}

#endif  // SYCLDNN_BENCH_BIAS_BASE_BIAS_FIXTURE_H_
