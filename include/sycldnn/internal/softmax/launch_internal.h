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

#ifndef SYCLDNN_INCLUDE_INTERNAL_SOFTMAX_LAUNCH_INTERNAL_H_
#define SYCLDNN_INCLUDE_INTERNAL_SOFTMAX_LAUNCH_INTERNAL_H_

#include "sycldnn/status.h"

#include "sycldnn/internal/pointwise/launch_internal.h"
#include "sycldnn/pointwise/direction.h"
#include "sycldnn/pointwise/operators.h"

namespace sycldnn {
namespace softmax {
namespace internal {

/**
 * The internal softmax launcher.
 *
 * Performs an element-wise exponentiation, followed by reduction
 * and then the pointwise division.
 */
template <typename T, typename SoftmaxType, typename Backend>
SNNStatus launch_softmax_forward(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T> workspace,
    typename Backend::template pointer_type<T> output,
    SoftmaxParams const& params, Backend& backend) {
  int32_t n_items = params.batch * params.channels * params.rows * params.cols;
  auto queue = backend.get_queue();
  auto in_mem = backend.get_mem_object(input, n_items);
  auto out_mem = backend.get_mem_object(output, n_items);
  int32_t workspace_items = params.batch * params.rows * params.cols;

  SNNStatus status = pointwise::internal::launch_pointwise<T, pointwise::Exp,
                                                           pointwise::Forward>(
      in_mem, out_mem, n_items, queue);

  using ConstPointer = typename Backend::template pointer_type<T const>;
  backend.template reduce<T, int32_t, softmax::SoftmaxParams>(
      ConstPointer{output}, workspace, params);

  auto const_workspace = ConstPointer{workspace};
  auto const_workspace_mem =
      backend.get_mem_object(const_workspace, workspace_items);

  status = pointwise::internal::launch_pointwise<T, pointwise::SoftMaxDiv,
                                                 pointwise::Forward>(
      const_workspace_mem, out_mem, params.channels, queue);
  return status;
}

}  // namespace internal
}  // namespace softmax
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_INTERNAL_SOFTMAX_LAUNCH_INTERNAL_H_