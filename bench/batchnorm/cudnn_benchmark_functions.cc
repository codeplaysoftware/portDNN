/*
 * Copyright Codeplay Software Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
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
#include "cudnn_batchnorm_bench_executor.h"
#include "sycldnn/format_type.h"

#define BENCHMARK_WITH_DTYPE_LAYOUT_BACKEND_DIRECTION_PROP(DTYPE, LAYOUT, DIR) \
  BATCHNORM_BENCHMARK(CUDNN##_##BACKEND##_##LAYOUT##_##DTYPE##_##DIR, DTYPE,   \
                      sycldnn::layout::LAYOUT, sycldnn::batchnorm::DIR)

#define Training true
#define Inference false

#define BENCHMARK_WITH_DTYPE_LAYOUT(DTYPE, LAYOUT)                     \
  BENCHMARK_WITH_DTYPE_LAYOUT_BACKEND_DIRECTION_PROP(DTYPE, LAYOUT,    \
                                                     ForwardTraining)  \
  BENCHMARK_WITH_DTYPE_LAYOUT_BACKEND_DIRECTION_PROP(DTYPE, LAYOUT,    \
                                                     ForwardInference) \
  BENCHMARK_WITH_DTYPE_LAYOUT_BACKEND_DIRECTION_PROP(DTYPE, LAYOUT, Gradient)

#ifdef SNN_ENABLE_NCHW
#define BENCHMARK_WITH_DTYPE(DTYPE)        \
  BENCHMARK_WITH_DTYPE_LAYOUT(DTYPE, NCHW) \
  BENCHMARK_WITH_DTYPE_LAYOUT(DTYPE, NHWC)
#else
#define BENCHMARK_WITH_DTYPE(DTYPE) BENCHMARK_WITH_DTYPE_LAYOUT(DTYPE, NHWC)
#endif

BENCHMARK_WITH_DTYPE(float)

#undef Training
#undef Inference
