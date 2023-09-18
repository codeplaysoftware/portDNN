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

#ifndef PORTDNN_SRC_POOLING_CAN_VECTORIZE_H_
#define PORTDNN_SRC_POOLING_CAN_VECTORIZE_H_

#include "portdnn/pooling/operators.h"
#include "portdnn/pooling/params.h"

namespace sycldnn {
namespace pooling {
namespace internal {

template <typename Direction, template <typename> class Op>
inline bool can_vectorize(PoolingParams const& pp, int width) {
  return (pp.input_format == DataFormat::NHWC) && ((pp.channels % width) == 0);
}

template <>
inline bool can_vectorize<Backpropagate, Max>(PoolingParams const&, int) {
  return false;
}

}  // namespace internal
}  // namespace pooling
}  // namespace sycldnn

#endif  // PORTDNN_SRC_POOLING_CAN_VECTORIZE_H_
