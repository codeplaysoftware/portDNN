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

#ifndef PORTDNN_INCLUDE_POINTWISE_LAUNCH_INTERNAL_H_
#define PORTDNN_INCLUDE_POINTWISE_LAUNCH_INTERNAL_H_

#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/pointwise/direction.h"
#include "portdnn/pointwise/operators.h"

#include "portdnn/export.h"

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
template <template <typename> class PointwiseType, typename T,
          typename Direction = Forward, template <typename> class MemObj,
          typename = DisableIfGradient<Direction>>
SNN_EXPORT SNNStatus launch_pointwise(
    MemObj<T const>& input, MemObj<T>& output, size_t const n_items,
    cl::sycl::queue& queue, const std::vector<cl::sycl::event>& events);

// The internal pointwise operation launcher for the backward pass.
template <template <typename> class PointwiseType, typename T,
          typename Direction = Forward, template <typename> class MemObj,
          typename = EnableIfGradient<Direction>>
SNN_EXPORT SNNStatus launch_pointwise(
    MemObj<T const>& input_forward, MemObj<T const>& input_backprop,
    MemObj<T>& output, size_t const n_items, cl::sycl::queue& queue,
    const std::vector<cl::sycl::event>& events);

template <typename T, template <typename> class PointwiseType,
          typename Direction, typename Backend,
          typename = internal::DisableIfGradient<Direction>>
SNNStatus sublaunch(typename Backend::template pointer_type<T const> input,
                    typename Backend::template pointer_type<T> output,
                    size_t const n_items, Backend& backend,
                    const std::vector<cl::sycl::event>& events) {
  SNN_VALIDATE_PARAM(n_items > 0, "The number of items must be positive.");

  auto inp_access = backend.get_mem_object(input, n_items);
  auto outp_access = backend.get_mem_object(output, n_items);

  auto queue = backend.get_queue();
  return internal::launch_pointwise<PointwiseType, T, Direction>(
      inp_access, outp_access, n_items, queue, events);
}

template <typename T, template <typename> class PointwiseType,
          typename Direction, typename Backend,
          typename = internal::EnableIfGradient<Direction>>
SNNStatus sublaunch(
    typename Backend::template pointer_type<T const> input_forward,
    typename Backend::template pointer_type<T const> input_backprop,
    typename Backend::template pointer_type<T> output_backprop,
    size_t const n_items, Backend& backend,
    const std::vector<cl::sycl::event>& events) {
  SNN_VALIDATE_PARAM(n_items > 0, "The number of items must be positive.");

  auto inp_fwd_access = backend.get_mem_object(input_forward, n_items);
  auto inp_bk_access = backend.get_mem_object(input_backprop, n_items);
  auto out_bk_access = backend.get_mem_object(output_backprop, n_items);

  auto queue = backend.get_queue();
  return internal::launch_pointwise<PointwiseType, T, Direction>(
      inp_fwd_access, inp_bk_access, out_bk_access, n_items, queue, events);
}

}  // namespace internal
}  // namespace pointwise
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_POINTWISE_LAUNCH_INTERNAL_H_
