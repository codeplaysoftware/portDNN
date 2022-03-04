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

char const* get_benchmark_name() { return "VGG"; }

// Note that the config order does not match the expected order for
// serialization.
// TODO(jwlawson): Unify pooling param ordering
#define CONFIG(N, C, H, W, WIN, STR, PAD) \
  benchmark_params::serialize(N, WIN, STR, H, W, C, PAD)

std::vector<std::vector<int>> const& get_benchmark_configs() {
  static std::vector<std::vector<int>> const configs = {

// Standard benchmark sizes (batch size: 1, 4, optionally 32
#define VGG_PARAMS(C, H, W, K, S, PAD) CONFIG(1, C, H, W, K, S, PAD),
#include "bench/pooling/vgg_params.def"
#undef VGG_PARAMS

#define VGG_PARAMS(C, H, W, K, S, PAD) CONFIG(4, C, H, W, K, S, PAD),
#include "bench/pooling/vgg_params.def"
#undef VGG_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define VGG_PARAMS(C, H, W, K, S, PAD) CONFIG(32, C, H, W, K, S, PAD),
#include "bench/pooling/vgg_params.def"
#undef VGG_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS

// Extended benchmarks (batch size: 2, optionally 8, 16, 64)
#ifdef SNN_EXTENDED_BENCHMARKS
#define VGG_PARAMS(C, H, W, K, S, PAD) CONFIG(2, C, H, W, K, S, PAD),
#include "bench/pooling/vgg_params.def"
#undef VGG_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define VGG_PARAMS(C, H, W, K, S, PAD) CONFIG(8, C, H, W, K, S, PAD),
#include "bench/pooling/vgg_params.def"
#undef VGG_PARAMS

#define VGG_PARAMS(C, H, W, K, S, PAD) CONFIG(16, C, H, W, K, S, PAD),
#include "bench/pooling/vgg_params.def"
#undef VGG_PARAMS

#define VGG_PARAMS(C, H, W, K, S, PAD) CONFIG(64, C, H, W, K, S, PAD),
#include "bench/pooling/vgg_params.def"
#undef VGG_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS
#endif  // SNN_EXTENDED_BENCHMARKS

  };
  return configs;
}
