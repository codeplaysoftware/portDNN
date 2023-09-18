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
#ifndef PORTDNN_BENCH_BATCHNORM_BASE_BATCHNORM_FIXTURE_H_
#define PORTDNN_BENCH_BATCHNORM_BASE_BATCHNORM_FIXTURE_H_

#include <benchmark/benchmark.h>

#include "portdnn/batchnorm/params.h"
#include "portdnn/helpers/dims.h"

extern const char* commit_date;
extern const char* commit_hash;

class BaseBatchnormBenchmark : public benchmark::Fixture {
 private:
  using State = benchmark::State;
  using Params = sycldnn::batchnorm::BatchNormParams;

 public:
  // Adds the batchnorm parameters to the counter set.
  void add_param_counters(State& state, Params const& params);

  // Adds theoretical best-case bandwidth requirements to the counter set.
  template <typename T>
  void add_bandwidth_counters(State& state, Params const& params);

  // Records the number of elements processed to the counter set. How this
  // calculated varies based on the type of operation.
  inline void set_items_processed(State& state, Params const& params);
};

// Add a full set of counters corresponding to the batchnorm parameters.
void BaseBatchnormBenchmark::add_param_counters(benchmark::State& state,
                                                Params const& params) {
  state.counters["batch"] = params.batch;
  state.counters["rows"] = params.rows;
  state.counters["cols"] = params.cols;
  state.counters["channels"] = params.channels;
}

// Calculate the optimal bandwidth requirements, and add corresponding counters.
// This assumes each batchnorm element is read exactly once, rather than the
// actual behaviour where multiple threads may re-read the same values.
template <typename ElementType>
void BaseBatchnormBenchmark::add_bandwidth_counters(benchmark::State& state,
                                                    Params const& params) {
  // Compute the size of each element in bytes.
  auto element_bytes = sizeof(ElementType);
  auto n_items = params.batch * params.rows * params.cols * params.channels;

  state.counters["bytes_read"] = n_items * element_bytes;
  state.counters["bytes_written"] = n_items * element_bytes;
}

// Records the number of elements processed to the counter set. How this
// is calculated varies based on the type of operation.
inline void BaseBatchnormBenchmark::set_items_processed(benchmark::State& state,
                                                        Params const& params) {
  // We define items processed as neighbourhood size * output tensor size for
  // batchnorm operations.
  auto n_items = params.batch * params.rows * params.cols * params.channels;

  state.SetItemsProcessed(state.iterations() * n_items);
}

#endif  // PORTDNN_BENCH_BATCHNORM_BASE_BATCHNORM_FIXTURE_H_
