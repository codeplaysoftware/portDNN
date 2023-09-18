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
#ifndef PORTDNN_BENCH_FIXTURE_STRING_REPORTER_H_
#define PORTDNN_BENCH_FIXTURE_STRING_REPORTER_H_

#include <benchmark/benchmark.h>

#include <map>
#include <string>

namespace sycldnn {
namespace bench {

/** Provide string labels to report for a benchmark. */
struct StringReporter {
 public:
  /**
   * Serialize the key-value map into a single comma separated string and
   * store it in the benchmark label.
   * */
  void set_label(::benchmark::State& state) {
    std::string label;
    for (auto& kv : key_value_map) {
      if (label.size()) {
        label += ",";
      }

      label += kv.first + "=" + kv.second;
    }
    state.SetLabel(label);
  }

  /**
   * Add a key-value pair to the label.
   *
   * Will overwrite the current value of key if one has already been set.
   */
  void add_to_label(std::string const& key, std::string const& value) {
    key_value_map[key] = value;
  }

 private:
  /** A map holding key-value pairs to be emitted along with the counter set. */
  std::map<std::string, std::string> key_value_map;
};

}  // namespace bench
}  // namespace sycldnn

#endif  // PORTDNN_BENCH_FIXTURE_STRING_REPORTER_H_
