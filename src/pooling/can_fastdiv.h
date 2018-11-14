/*
 * Copyright 2018 Codeplay Software Ltd
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

#ifndef SYCLDNN_SRC_POOLING_CAN_FASTDIV_H_
#define SYCLDNN_SRC_POOLING_CAN_FASTDIV_H_

#include "sycldnn/pooling/operators.h"
#include "sycldnn/pooling/params.h"

namespace sycldnn {
namespace pooling {
namespace internal {

template <typename Direction>
inline bool can_use_fastdiv(PoolingParams const& pp, int vector_width) {
  return !((pp.channels / vector_width) == 1 || pp.out_rows == 1 ||
           pp.out_cols == 1);
}

template <>
inline bool can_use_fastdiv<Backpropagate>(PoolingParams const& pp,
                                           int vector_width) {
  return !((pp.channels / vector_width) == 1 || pp.in_rows == 1 ||
           pp.in_cols == 1);
}

}  // namespace internal
}  // namespace pooling
}  // namespace sycldnn
#endif  // SYCLDNN_SRC_POOLING_CAN_FASTDIV_H_
