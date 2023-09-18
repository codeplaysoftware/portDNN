/*
 * Copyright Codeplay Software Ltd.
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
#include "snn_fixture.h"

#include "src/backend/snn_backend_provider.h"

#include "portdnn/backend/snn_backend.h"

#include "portdnn/pointwise/direction.h"
#include "portdnn/pointwise/operators.h"

#define RELU_BM_WITH_DIRECTION_AND_DTYPE(N, DIRECTION, DTYPE)        \
  POINTWISE_BENCHMARK("Relu", OP##_##DIRECTION##_##N##_##SNNBackend, \
                      sycldnn::backend::SNNBackend, DTYPE, N,        \
                      sycldnn::pointwise::DIRECTION, sycldnn::pointwise::Relu)

#define RELU_BM_WITH_DIRECTION(N, DIRECTION) \
  RELU_BM_WITH_DIRECTION_AND_DTYPE(N, DIRECTION, float)

#define RELU_BENCHMARK(N)            \
  RELU_BM_WITH_DIRECTION(N, Forward) \
  RELU_BM_WITH_DIRECTION(N, Gradient)

/** Sizes used correspond to the sizes of inputs for relu layers in ResNet.
 * Where the resulting sizes are identical, they are skipped.
 *
 * Channels | Width | Height |
 * ---------|-------|--------|
 *       64 |   112 |    112 | --> 802,816
 *       64 |    56 |     56 | --> 200,704
 *      128 |    28 |     28 | --> 100,352
 *      512 |    28 |     28 | --> 401,408
 *      256 |    14 |     14 | -->  50,176
 */
RELU_BENCHMARK(802816);
RELU_BENCHMARK(200704);
RELU_BENCHMARK(100352);
RELU_BENCHMARK(401408);
RELU_BENCHMARK(50176);
