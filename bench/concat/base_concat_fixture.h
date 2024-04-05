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
#ifndef SYCLDNN_BENCH_CONCAT_BASE_CONCAT_FIXTURE_H_
#define SYCLDNN_BENCH_CONCAT_BASE_CONCAT_FIXTURE_H_

#include <benchmark/benchmark.h>

#include <numeric>

#include "sycldnn/concat/launch.h"
#include "sycldnn/concat/params.h"

extern const char* commit_date;
extern const char* commit_hash;

class BaseConcatBenchmark : public benchmark::Fixture {
 private:
  using State = benchmark::State;

 public:
  /** Adds the number of elements and tensor rank to the counter set. */
  void add_param_counters(benchmark::State& state,
                          sycldnn::concat::ConcatParams& params);

  /** Adds theoretical best-case bandwidth requirements to the counter set. */
  template <typename T>
  void add_bandwidth_counters(State& state, size_t const n_items);

  /** Records the number of elements processed to the counter set. */
  template <typename T>
  void set_bytes_processed(State& state, size_t const n_items);
};

/** Add counters corresponding to the number of tensors, the overall number of
 * items in all of the tensors and the tensor rank. */
void BaseConcatBenchmark::add_param_counters(
    benchmark::State& state, sycldnn::concat::ConcatParams& params) {
  state.counters["n_tensors"] = params.input_descriptors.size();
  state.counters["n_items"] =
      sycldnn::concat::internal::get_concat_output_size(params);
  state.counters["rank"] = params.rank;
  state.counters["axis"] = params.axis;

  std::vector<int> flat_sizes;
  for (auto desc : params.input_descriptors) {
    flat_sizes.push_back(
        std::accumulate(desc.begin(), desc.end(), 1, std::multiplies<int>()));
  }
  if (std::equal(flat_sizes.begin() + 1, flat_sizes.end(),
                 flat_sizes.begin())) {
    state.counters["tensor_size"] = flat_sizes[0];
  } else {
    for (size_t i = 0; i < params.input_descriptors.size(); i++) {
      state.counters["tensor" + std::to_string(i) + "_size"] = flat_sizes[i];
    }
  }
}

/** Calculate the optimal bandwidth requirements, and add corresponding
 * counters. This assumes each input element is read exactly once, rather than
 * the actual behaviour where multiple threads may re-read the same values. */
template <typename T>
void BaseConcatBenchmark::add_bandwidth_counters(benchmark::State& state,
                                                 size_t const n_items) {
  auto element_bytes = sizeof(T);
  state.counters["bytes_read"] = n_items * element_bytes;
  state.counters["bytes_written"] = n_items * element_bytes;
}

/** For concat we read from one variable and write out to one, giving three
 * memory accesses. */
template <typename T>
void BaseConcatBenchmark::set_bytes_processed(benchmark::State& state,
                                              size_t const n_items) {
  auto element_bytes = sizeof(T);
  state.SetBytesProcessed(state.iterations() * n_items * 2 * element_bytes);
}

#endif
