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

#include "portdnn/padding_mode.h"

#include <vector>

char const* get_benchmark_name() { return "SimpleConvolution"; }

#define CONFIG(N, WIN, STR, H, W, C, F, MOD) \
  benchmark_params::serialize(N, WIN, STR, H, W, C, F, MOD)

std::vector<std::vector<int>> const& get_benchmark_configs() {
  static std::vector<std::vector<int>> const configs = {
      CONFIG(1, 3, 1, 27, 27, 196, 384, sycldnn::PaddingMode::SAME),
      CONFIG(1, 3, 2, 27, 27, 196, 384, sycldnn::PaddingMode::SAME),
  };
  return configs;
}
