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
#ifndef SYCLDNN_BENCH_REDUCTION_BASE_REDUCTION_FIXTURE_H_
#define SYCLDNN_BENCH_REDUCTION_BASE_REDUCTION_FIXTURE_H_

#include <benchmark/benchmark.h>

#include "sycldnn/helpers/dims.h"

extern const char* commit_date;
extern const char* commit_hash;

class BaseReductionBenchmark : public benchmark::Fixture {
 private:
  using State = benchmark::State;

 public:
  // Adds the reduction parameters to the counter set.
  void add_param_counters(State& state, int, int, int);

  // Adds theoretical best-case bandwidth requirements to the counter set.
  template <typename T>
  void add_bandwidth_counters(State& state, int, int, int);

  // Records the number of elements processed to the counter set. How this is
  // calculated varies based on the type of operation.
  inline void set_items_processed(State& state, int, int, int);
};

// Add a full set of counters corresponding to the reduction parameters.
void BaseReductionBenchmark::add_param_counters(benchmark::State& state,
                                                int batch, int outer,
                                                int inner) {
  state.counters["batch"] = batch;
  state.counters["outer"] = outer;
  state.counters["inner"] = inner;
}

// Calculate the optimal bandwidth requirements, and add corresponding counters.
// This assumes each reduction element is read exactly once, rather than the
// actual behaviour where multiple threads may re-read the same values.
template <typename ElementType>
void BaseReductionBenchmark::add_bandwidth_counters(benchmark::State& state,
                                                    int batch, int outer,
                                                    int inner) {
  // Compute the size of each element in bytes.
  auto element_bytes = sizeof(ElementType);
  auto n_items = batch * outer * inner;

  state.counters["bytes_read"] = n_items * element_bytes;
  state.counters["bytes_written"] = n_items * element_bytes;
}

// Records the number of elements processed to the counter set. How this
// is calculated varies based on the type of operation.
inline void BaseReductionBenchmark::set_items_processed(benchmark::State& state,
                                                        int batch, int outer,
                                                        int inner) {
  auto n_items = batch * outer * inner;

  state.SetItemsProcessed(state.iterations() * n_items);
}

#endif  // SYCLDNN_BENCH_REDUCTION_BASE_REDUCTION_FIXTURE_H_
