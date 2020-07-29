/*
 * Copyright Codeplay Software Ltd
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

char const* matmul_benchmark_name() { return "Resnet Network Matmuls"; }

#define RESNET_PARAMS(M, K, N, BATCH, TRANS_L, TRANS_R) \
  matmul_benchmark_params::serialize(M, K, N, BATCH, TRANS_L, TRANS_R),

std::vector<std::vector<int>> const& matmul_benchmark_configs() {
  static std::vector<std::vector<int>> const configs = {
#include "bench/matmul/resnet_params.def"
  };
  return configs;
}
