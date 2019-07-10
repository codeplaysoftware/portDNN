/*
 * Copyright 2019 Codeplay Software Ltd.
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
#include "param_set.h"
#include "snn_fixture.h"

#include "src/backend/snn_backend_provider.h"

#include "sycldnn/backend/snn_backend.h"

#include "sycldnn/conv2d/conv_type.h"

#define XCEPTION_BENCHMARK_WITH_DIR_DTYPE(N, WIN, STR, H, W, C, MUL, PAD, Dir, \
                                          DType)                               \
  DEPTHWISE_CONVOLUTION_BENCHMARK(                                             \
      "Xception", Dir##_##N##_##WIN##_##STR##_##H##_##W##_##C,                 \
      sycldnn::backend::SNNBackend, DType,                                     \
      ParameterSet<N, WIN, STR, H, W, C, MUL, PAD>,                            \
      sycldnn::conv2d::conv_type::Dir)

#define XCEPTION_BENCHMARK_WITH_DIR(N, WIN, STR, H, W, C, MUL, PAD, Dir) \
  XCEPTION_BENCHMARK_WITH_DIR_DTYPE(N, WIN, STR, H, W, C, MUL, PAD, Dir, float)

#define XCEPTION_BENCHMARK(N, WIN, STR, H, W, C, MUL, PAD)                   \
  XCEPTION_BENCHMARK_WITH_DIR(N, WIN, STR, H, W, C, MUL, PAD, Forward)       \
  XCEPTION_BENCHMARK_WITH_DIR(N, WIN, STR, H, W, C, MUL, PAD, InputBackprop) \
  XCEPTION_BENCHMARK_WITH_DIR(N, WIN, STR, H, W, C, MUL, PAD, FilterBackprop)

// Standard benchmark sizes (batch size: 1, 4, optionally 32
#define XCEPTION_PARAMS(WIN, STR, H, W, C, MUL, PAD) \
  XCEPTION_BENCHMARK(1, WIN, STR, H, W, C, MUL, PAD);
#include "bench/depthwise_conv2d/xception_params.def"
#undef XCEPTION_PARAMS

#define XCEPTION_PARAMS(WIN, STR, H, W, C, MUL, PAD) \
  XCEPTION_BENCHMARK(4, WIN, STR, H, W, C, MUL, PAD);
#include "bench/depthwise_conv2d/xception_params.def"
#undef XCEPTION_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define XCEPTION_PARAMS(WIN, STR, H, W, C, MUL, PAD) \
  XCEPTION_BENCHMARK(32, WIN, STR, H, W, C, MUL, PAD);
#include "bench/depthwise_conv2d/xception_params.def"
#undef XCEPTION_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS

// Extended benchmarks (batch size: 2, optionally 8, 16, 64)
#ifdef SNN_EXTENDED_BENCHMARKS
#define XCEPTION_PARAMS(WIN, STR, H, W, C, MUL, PAD) \
  XCEPTION_BENCHMARK(2, WIN, STR, H, W, C, MUL, PAD);
#include "bench/depthwise_conv2d/xception_params.def"
#undef XCEPTION_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define XCEPTION_PARAMS(WIN, STR, H, W, C, MUL, PAD) \
  XCEPTION_BENCHMARK(8, WIN, STR, H, W, C, MUL, PAD);
#include "bench/depthwise_conv2d/xception_params.def"
#undef XCEPTION_PARAMS

#define XCEPTION_PARAMS(WIN, STR, H, W, C, MUL, PAD) \
  XCEPTION_BENCHMARK(16, WIN, STR, H, W, C, MUL, PAD);
#include "bench/depthwise_conv2d/xception_params.def"
#undef XCEPTION_PARAMS

#define XCEPTION_PARAMS(WIN, STR, H, W, C, MUL, PAD) \
  XCEPTION_BENCHMARK(64, WIN, STR, H, W, C, MUL, PAD);
#include "bench/depthwise_conv2d/xception_params.def"
#undef XCEPTION_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS
#endif  // SNN_EXTENDED_BENCHMARKS
