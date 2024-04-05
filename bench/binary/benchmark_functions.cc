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

#include "sycldnn/binaryop/operators.h"

#define BM_WITH_OP_DTYPE(OP, DTYPE)                                        \
  BINARY_BENCHMARK(OP##_##SNNBackend, sycldnn::backend::SNNBackend, DTYPE, \
                   sycldnn::binaryop::OP)

#define BM_WITH_OP(OP) BM_WITH_OP_DTYPE(OP, float)

BM_WITH_OP(Add);
