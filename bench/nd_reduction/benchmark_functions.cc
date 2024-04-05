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
#include "src/backend/snn_usm_backend_provider.h"

#include "sycldnn/backend/snn_backend.h"
#include "sycldnn/reduce/operators.h"

#define BM_WITH_DTYPE_OP(DTYPE, BACKEND, OP)                               \
  ND_REDUCTION_BENCHMARK(OP##_##BACKEND, sycldnn::backend::BACKEND, DTYPE, \
                         sycldnn::reduce::OP)

#define BM_WITH_BACKEND(DTYPE, OP)        \
  BM_WITH_DTYPE_OP(DTYPE, SNNBackend, OP) \
  BM_WITH_DTYPE_OP(DTYPE, SNNUSMBackend, OP)

#define BM_WITH_OP(OP) BM_WITH_BACKEND(float, OP)

BM_WITH_OP(Add)
BM_WITH_OP(Mean)
BM_WITH_OP(Max)
BM_WITH_OP(Min)
