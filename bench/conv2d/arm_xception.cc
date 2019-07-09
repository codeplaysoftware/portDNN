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
#if defined(ARM_COMPUTE)
#include "arm_fixture.h"
// For ARM Compute Library, need to provide the executor to specify whether
// running on NEON or OpenCL.
#ifdef ACL_NEON
#define EXEC sycldnn::bench::ACLNeonExecutor
#else
#define EXEC sycldnn::bench::ACLOpenCLExecutor
#endif  // ACL_NEON

#elif defined(MKL_DNN)
#include "mkldnn_conv2d_executor.h"
// For MKL-DNN, there is currently only one Executor which uses the CPU, so for
// now pass in a dummy value.
struct Executor {};
#define EXEC Executor

#else
#error Cannot compile without either ARM_COMPUTE or MKL_DNN defined
#endif

#include "param_set.h"

#define XCEPTION_BENCHMARK_WITH_DTYPE(N, WIN, STR, H, W, C, F, MOD, DTYPE)    \
  CONVOLUTION_BENCHMARK(                                                      \
      "Xception", Forward_##N##_##WIN##_##STR##_##H##_##W##_##C##_##F, DTYPE, \
      ParameterSet<N, WIN, STR, H, W, C, F, MOD>, EXEC)

#define XCEPTION_BENCHMARK(N, WIN, STR, H, W, C, F, MOD) \
  XCEPTION_BENCHMARK_WITH_DTYPE(N, WIN, STR, H, W, C, F, MOD, float)

// Standard benchmark sizes (batch size: 1, 4, optionally 32)
#define XCEPTION_PARAMS(WIN, STR, H, W, C, F, MOD) \
  XCEPTION_BENCHMARK(1, WIN, STR, H, W, C, F, MOD);
#include "bench/conv2d/xception_params.def"
#undef XCEPTION_PARAMS

#define XCEPTION_PARAMS(WIN, STR, H, W, C, F, MOD) \
  XCEPTION_BENCHMARK(4, WIN, STR, H, W, C, F, MOD);
#include "bench/conv2d/xception_params.def"
#undef XCEPTION_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define XCEPTION_PARAMS(WIN, STR, H, W, C, F, MOD) \
  XCEPTION_BENCHMARK(32, WIN, STR, H, W, C, F, MOD);
#include "bench/conv2d/xception_params.def"
#undef XCEPTION_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS

// Extended benchmarks (batch size: 2, optionally 8, 16, 64)
#ifdef SNN_EXTENDED_BENCHMARKS
#define XCEPTION_PARAMS(WIN, STR, H, W, C, F, MOD) \
  XCEPTION_BENCHMARK(2, WIN, STR, H, W, C, F, MOD);
#include "bench/conv2d/xception_params.def"
#undef XCEPTION_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define XCEPTION_PARAMS(WIN, STR, H, W, C, F, MOD) \
  XCEPTION_BENCHMARK(8, WIN, STR, H, W, C, F, MOD);
#include "bench/conv2d/xception_params.def"
#undef XCEPTION_PARAMS

#define XCEPTION_PARAMS(WIN, STR, H, W, C, F, MOD) \
  XCEPTION_BENCHMARK(16, WIN, STR, H, W, C, F, MOD);
#include "bench/conv2d/xception_params.def"
#undef XCEPTION_PARAMS

#define XCEPTION_PARAMS(WIN, STR, H, W, C, F, MOD) \
  XCEPTION_BENCHMARK(64, WIN, STR, H, W, C, F, MOD);
#include "bench/conv2d/xception_params.def"
#undef XCEPTION_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS
#endif  // SNN_EXTENDED_BENCHMARKS
