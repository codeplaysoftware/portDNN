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

#include "portdnn/backend/snn_backend.h"

#include "portdnn/pooling/operators.h"

#define BM_WITH_DIR_OP_DTYPE(DIR, OP, DTYPE)                                 \
  POOLING_BENCHMARK(OP##_##DIR##_##SNNBackend, sycldnn::backend::SNNBackend, \
                    DTYPE, sycldnn::pooling::DIR, sycldnn::pooling::OP)

#define BM_WITH_DIR_OP(DIR, OP) BM_WITH_DIR_OP_DTYPE(DIR, OP, float)

#define BM_WITH_DIRECTION(DIR) \
  BM_WITH_DIR_OP(DIR, Max)     \
  BM_WITH_DIR_OP(DIR, Average)

BM_WITH_DIRECTION(Forward);
BM_WITH_DIRECTION(Backpropagate);
