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
#include "benchmark_config.h"
#include "benchmark_params.h"

#include <vector>

char const* get_benchmark_name() { return "reductions"; }

// Note that the config order does not match the expected order for
// serialization.
#define CONFIG(D1, D2, D3, D4, A1, A2) \
  benchmark_params::serialize({D1, D2, D3, D4}, {A1, A2})

std::vector<std::vector<int>> const& get_benchmark_configs() {
  static std::vector<std::vector<int>> const configs = {
      CONFIG(1, 65536, 1, 1, 1, 2),    CONFIG(2, 65536, 1, 1, 1, 2),
      CONFIG(8, 65536, 1, 1, 1, 2),    CONFIG(32, 65536, 1, 1, 1, 2),
      CONFIG(128, 65536, 1, 1, 1, 2),  CONFIG(1, 65536, 2, 1, 1, 2),
      CONFIG(2, 65536, 2, 1, 1, 2),    CONFIG(8, 65536, 2, 1, 1, 2),
      CONFIG(32, 65536, 2, 1, 1, 2),   CONFIG(128, 65536, 2, 1, 1, 2),
      CONFIG(1, 65536, 8, 1, 1, 2),    CONFIG(2, 65536, 8, 1, 1, 2),
      CONFIG(8, 65536, 8, 1, 1, 2),    CONFIG(32, 65536, 8, 1, 1, 2),
      CONFIG(128, 65536, 8, 1, 1, 2),  CONFIG(1, 65536, 32, 1, 1, 2),
      CONFIG(2, 65536, 32, 1, 1, 2),   CONFIG(8, 65536, 32, 1, 1, 2),
      CONFIG(32, 65536, 32, 1, 1, 2),  CONFIG(128, 65536, 32, 1, 1, 2),
      CONFIG(1, 65536, 128, 1, 1, 2),  CONFIG(2, 65536, 128, 1, 1, 2),
      CONFIG(8, 65536, 128, 1, 1, 2),  CONFIG(32, 65536, 128, 1, 1, 2),
      CONFIG(128, 65536, 128, 1, 1, 2)};
  return configs;
}
