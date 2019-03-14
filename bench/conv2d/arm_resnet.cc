/*
 * Copyright 2018 Codeplay Software Ltd.
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
#include "arm_fixture.h"
#include "resnet_param_set.h"

#ifdef ACL_NEON
#define EXEC sycldnn::bench::ACLNeonExecutor
#else
#define EXEC sycldnn::bench::ACLOpenCLExecutor
#endif

#define RESNET_BENCHMARK(N, C, W, H, Flt, S, Ftr)                        \
  CONVOLUTION_BENCHMARK(                                                 \
      "ResNet", ARM_Forward_##N##_##C##_##W##_##H##_##Flt##_##S##_##Ftr, \
      ParameterSet<N, C, W, H, Flt, S, Ftr>, EXEC)

// Standard benchmark sizes (batch size: 1, 4, optionally 32
#define RESNET_PARAMS(channels, width, height, filter, stride, features) \
  RESNET_BENCHMARK(1, channels, width, height, filter, stride, features);
#include "bench/conv2d/resnet_params.def"
#undef RESNET_PARAMS

#define RESNET_PARAMS(channels, width, height, filter, stride, features) \
  RESNET_BENCHMARK(4, channels, width, height, filter, stride, features);
#include "bench/conv2d/resnet_params.def"
#undef RESNET_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define RESNET_PARAMS(channels, width, height, filter, stride, features) \
  RESNET_BENCHMARK(32, channels, width, height, filter, stride, features);
#include "bench/conv2d/resnet_params.def"
#undef RESNET_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS

// Extended benchmarks (batch size: 2, optionally 8, 16, 64)
#ifdef SNN_EXTENDED_BENCHMARKS
#define RESNET_PARAMS(channels, width, height, filter, stride, features) \
  RESNET_BENCHMARK(2, channels, width, height, filter, stride, features);
#include "bench/conv2d/resnet_params.def"
#undef RESNET_PARAMS

#ifdef SNN_LARGE_BATCH_BENCHMARKS
#define RESNET_PARAMS(channels, width, height, filter, stride, features) \
  RESNET_BENCHMARK(8, channels, width, height, filter, stride, features);
#include "bench/conv2d/resnet_params.def"
#undef RESNET_PARAMS

#define RESNET_PARAMS(channels, width, height, filter, stride, features) \
  RESNET_BENCHMARK(16, channels, width, height, filter, stride, features);
#include "bench/conv2d/resnet_params.def"
#undef RESNET_PARAMS

#define RESNET_PARAMS(channels, width, height, filter, stride, features) \
  RESNET_BENCHMARK(64, channels, width, height, filter, stride, features);
#include "bench/conv2d/resnet_params.def"
#undef RESNET_PARAMS
#endif  // SNN_LARGE_BATCH_BENCHMARKS
#endif  // SNN_EXTENDED_BENCHMARKS
