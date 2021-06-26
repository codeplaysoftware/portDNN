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

char const* get_benchmark_name() { return "Net"; }

// Note that the config order does not match the expected order for
// serialization.
#define CONFIG(N, C, H, W, B) benchmark_params::serialize(N, H, W, C, B)

std::vector<std::vector<int>> const& get_benchmark_configs() {
  static std::vector<std::vector<int>> const configs = {

// Standard benchmark sizes (batch size: 1, 4, optionally 32
#define NET_PARAMS(C, H, W, B) CONFIG(1, C, H, W, B),
#include "bench/bias/net_params.def"
#undef NET_PARAMS

#define NET_PARAMS(C, H, W, B) CONFIG(4, C, H, W, B),
#include "bench/bias/net_params.def"
#undef NET_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define NET_PARAMS(C, H, W, B) CONFIG(32, C, H, W, B),
#include "bench/bias/net_params.def"
#undef NET_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS

// Extended benchmarks (batch size: 2, optionally 8, 16, 64)
#ifdef SNN_EXTENDED_BENCHMARKS
#define NET_PARAMS(C, H, W, B) CONFIG(2, C, H, W, B),
#include "bench/bias/net_params.def"
#undef NET_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define NET_PARAMS(C, H, W, B) CONFIG(8, C, H, W, B),
#include "bench/bias/net_params.def"
#undef NET_PARAMS

#define NET_PARAMS(C, H, W, B) CONFIG(16, C, H, W, B),
#include "bench/bias/net_params.def"
#undef NET_PARAMS

#define NET_PARAMS(C, H, W, B) CONFIG(64, C, H, W, B),
#include "bench/bias/net_params.def"
#undef NET_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS
#endif  // SNN_EXTENDED_BENCHMARKS

  };
  return configs;
}
