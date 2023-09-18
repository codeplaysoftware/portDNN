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
#ifndef PORTDNN_BENCH_POINTWISE_BASE_POINTWISE_FIXTURE_H_
#define PORTDNN_BENCH_POINTWISE_BASE_POINTWISE_FIXTURE_H_

#include <benchmark/benchmark.h>

#include "portdnn/pointwise/direction.h"
#include "portdnn/pointwise/operators.h"

extern const char* commit_date;
extern const char* commit_hash;

class BasePointwiseBenchmark : public benchmark::Fixture {
 private:
  using State = benchmark::State;

 public:
  /** Adds the number of elements to the counter set. */
  void add_param_counters(State& state, size_t const n_items);

  /** Adds theoretical best-case bandwidth requirements to the counter set. */
  template <typename T, typename Direction>
  void add_bandwidth_counters(State& state, size_t const n_items);

  /** Records the number of elements processed to the counter set. How this
   * is calculated varies based on the type of operation. */
  template <typename T>
  void set_bytes_processed(State& state, size_t const n_items);
};

/** Add a counter corresponding to the number of items in the input. */
void BasePointwiseBenchmark::add_param_counters(benchmark::State& state,
                                                size_t const n_items) {
  state.counters["n_items"] = n_items;
}

/** Calculate the optimal bandwidth requirements, and add corresponding
 * counters. This assumes each input element is read exactly once, rather than
 * the actual behaviour where multiple threads may re-read the same values. */
template <typename T, typename Direction>
void BasePointwiseBenchmark::add_bandwidth_counters(benchmark::State& state,
                                                    size_t const n_items) {
  auto element_bytes = sizeof(T);
  state.counters["bytes_read"] = n_items * element_bytes;
  state.counters["bytes_written"] = n_items * element_bytes;
}

/** For activation functions we read from two variables and write out
 * to one, giving three memory accesses.
 */
template <typename T>
void BasePointwiseBenchmark::set_bytes_processed(benchmark::State& state,
                                                 size_t const n_items) {
  auto element_bytes = sizeof(T);
  state.SetBytesProcessed(state.iterations() * n_items * 3 * element_bytes);
}

#endif  // PORTDNN_BENCH_POINTWISE_BASE_POINTWISE_FIXTURE_H_
