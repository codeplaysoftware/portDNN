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

#include "sycldnn/pooling/operators.h"

#define BM_WITH_DIR_OP_DTYPE_LAYOUT_BACKEND(DIR, OP, DTYPE, LAYOUT, BACKEND)   \
  POOLING_BENCHMARK(LAYOUT##_##OP##_##DIR##_##BACKEND##_##OutOfOrder,          \
                    sycldnn::backend::BACKEND, DTYPE, sycldnn::layout::LAYOUT, \
                    sycldnn::pooling::DIR, sycldnn::pooling::OP, false)        \
  POOLING_BENCHMARK(LAYOUT##_##OP##_##DIR##_##BACKEND##_##InOrder,             \
                    sycldnn::backend::BACKEND, DTYPE, sycldnn::layout::LAYOUT, \
                    sycldnn::pooling::DIR, sycldnn::pooling::OP, true)

#ifdef SNN_ENABLE_USM
#define BM_WITH_DIR_OP_DTYPE_LAYOUT(DIR, OP, DTYPE, LAYOUT)               \
  BM_WITH_DIR_OP_DTYPE_LAYOUT_BACKEND(DIR, OP, DTYPE, LAYOUT, SNNBackend) \
  BM_WITH_DIR_OP_DTYPE_LAYOUT_BACKEND(DIR, OP, DTYPE, LAYOUT, SNNUSMBackend)
#else
#define BM_WITH_DIR_OP_DTYPE_LAYOUT(DIR, OP, DTYPE, LAYOUT, BACKEND) \
  BM_WITH_DIR_OP_DTYPE_LAYOUT_BACKEND(DIR, OP, DTYPE, LAYOUT, SNNBackend)
#endif

#ifdef SNN_ENABLE_NCHW
#define BM_WITH_DIR_OP_DTYPE(DIR, OP, DTYPE)        \
  BM_WITH_DIR_OP_DTYPE_LAYOUT(DIR, OP, DTYPE, NHWC) \
  BM_WITH_DIR_OP_DTYPE_LAYOUT(DIR, OP, DTYPE, NCHW)
#else
#define BM_WITH_DIR_OP_DTYPE(DIR, OP, DTYPE) \
  BM_WITH_DIR_OP_DTYPE_LAYOUT(DIR, OP, DTYPE, NHWC)
#endif

#define BM_WITH_DIR_OP(DIR, OP) BM_WITH_DIR_OP_DTYPE(DIR, OP, float)

#define BM_WITH_DIRECTION(DIR) \
  BM_WITH_DIR_OP(DIR, Max)     \
  BM_WITH_DIR_OP(DIR, Average)

BM_WITH_DIRECTION(Forward);
BM_WITH_DIRECTION(Backpropagate);
