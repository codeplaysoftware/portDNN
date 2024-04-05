/*
 * Copyright Codeplay Software Ltd.
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
#include "snn_fixture.h"

#include "src/backend/snn_backend_provider.h"

#include "sycldnn/backend/snn_backend.h"
#include "sycldnn/backend/snn_usm_backend.h"
#include "sycldnn/batchnorm/direction.h"
#include "sycldnn/format_type.h"

#define BENCHMARK_WITH_DTYPE_LAYOUT_BACKEND_DIRECTION(DTYPE, LAYOUT, BACKEND, \
                                                      DIR)                    \
  BATCHNORM_BENCHMARK(BACKEND##_##LAYOUT##_##DTYPE##_##DIR##_InOrder,         \
                      sycldnn::backend::BACKEND, DTYPE,                       \
                      sycldnn::layout::LAYOUT, sycldnn::batchnorm::DIR,       \
                      /*InOrderQueue*/ true)                                  \
  BATCHNORM_BENCHMARK(BACKEND##_##LAYOUT##_##DTYPE##_##DIR##_OutOfOrder,      \
                      sycldnn::backend::BACKEND, DTYPE,                       \
                      sycldnn::layout::LAYOUT, sycldnn::batchnorm::DIR,       \
                      /*InOrderQueue*/ false);

#ifdef SNN_ENABLE_USM
#define BENCHMARK_WITH_DTYPE_LAYOUT(DTYPE, LAYOUT)                            \
  BENCHMARK_WITH_DTYPE_LAYOUT_BACKEND_DIRECTION(DTYPE, LAYOUT, SNNBackend,    \
                                                ForwardTraining)              \
  BENCHMARK_WITH_DTYPE_LAYOUT_BACKEND_DIRECTION(DTYPE, LAYOUT, SNNBackend,    \
                                                ForwardInference)             \
  BENCHMARK_WITH_DTYPE_LAYOUT_BACKEND_DIRECTION(DTYPE, LAYOUT, SNNBackend,    \
                                                Gradient)                     \
  BENCHMARK_WITH_DTYPE_LAYOUT_BACKEND_DIRECTION(DTYPE, LAYOUT, SNNUSMBackend, \
                                                ForwardTraining)              \
  BENCHMARK_WITH_DTYPE_LAYOUT_BACKEND_DIRECTION(DTYPE, LAYOUT, SNNUSMBackend, \
                                                ForwardInference)             \
  BENCHMARK_WITH_DTYPE_LAYOUT_BACKEND_DIRECTION(DTYPE, LAYOUT, SNNUSMBackend, \
                                                Gradient)
#else
#define BENCHMARK_WITH_DTYPE_LAYOUT(DTYPE, LAYOUT)                         \
  BENCHMARK_WITH_DTYPE_LAYOUT_BACKEND_DIRECTION(DTYPE, LAYOUT, SNNBackend, \
                                                ForwardTraining)           \
  BENCHMARK_WITH_DTYPE_LAYOUT_BACKEND_DIRECTION(DTYPE, LAYOUT, SNNBackend, \
                                                ForwardInference)          \
  BENCHMARK_WITH_DTYPE_LAYOUT_BACKEND_DIRECTION(DTYPE, LAYOUT, SNNBackend, \
                                                Gradient)
#endif

#ifdef SNN_ENABLE_NCHW
#define BENCHMARK_WITH_DTYPE(DTYPE)        \
  BENCHMARK_WITH_DTYPE_LAYOUT(DTYPE, NCHW) \
  BENCHMARK_WITH_DTYPE_LAYOUT(DTYPE, NHWC)
#else
#define BENCHMARK_WITH_DTYPE(DTYPE) BENCHMARK_WITH_DTYPE_LAYOUT(DTYPE, NHWC)
#endif

BENCHMARK_WITH_DTYPE(float)
