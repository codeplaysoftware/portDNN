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

#ifndef SYCLDNN_INCLUDE_POINTWISE_LAUNCH_INTERNAL_H_
#define SYCLDNN_INCLUDE_POINTWISE_LAUNCH_INTERNAL_H_

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/pointwise/direction.h"
#include "sycldnn/pointwise/operators.h"

namespace sycldnn {
namespace pointwise {
namespace internal {

template <typename Direction>
struct IsGradient {
  static constexpr bool value = std::is_same<Direction, Gradient>::value;
};

template <typename Direction>
using EnableIfGradient =
    typename std::enable_if<IsGradient<Direction>::value>::type;

template <typename Direction>
using DisableIfGradient =
    typename std::enable_if<!IsGradient<Direction>::value>::type;

// The internal pointwise operation launcher for the forward pass.
template <typename T, template <typename> class PointwiseType,
          typename Direction, typename = DisableIfGradient<Direction>>
SNNStatus launch_pointwise(BaseMemObject<T const>& input,
                           BaseMemObject<T>& output, size_t const n_items,
                           cl::sycl::queue& queue);

// The internal pointwise operation launcher for the backward pass.
template <typename T, template <typename> class PointwiseType,
          typename Direction, typename = EnableIfGradient<Direction>>
SNNStatus launch_pointwise(BaseMemObject<T const>& input_forward,
                           BaseMemObject<T const>& input_backprop,
                           BaseMemObject<T>& output, size_t const n_items,
                           cl::sycl::queue& queue);

}  // namespace internal
}  // namespace pointwise
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_POINTWISE_LAUNCH_INTERNAL_H_
