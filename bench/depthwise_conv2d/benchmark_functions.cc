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

#include "portdnn/conv2d/conv_type.h"

#define BM_WITH_DIR_DTYPE(DIR, DTYPE)                                       \
  DEPTHWISE_CONVOLUTION_BENCHMARK(DIR, sycldnn::backend::SNNBackend, DTYPE, \
                                  sycldnn::conv2d::conv_type::DIR)

#define BM_WITH_DIR(DIR) BM_WITH_DIR_DTYPE(DIR, float)

BM_WITH_DIR(Forward);
BM_WITH_DIR(InputBackprop);
BM_WITH_DIR(FilterBackprop);
