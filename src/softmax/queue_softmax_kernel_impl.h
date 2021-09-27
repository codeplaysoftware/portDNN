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

#ifndef SYCLDNN_SRC_SOFTMAX_QUEUE_IMPL_H
#define SYCLDNN_SRC_SOFTMAX_QUEUE_IMPL_H

#include "sycldnn/status.h"

#include "sycldnn/softmax/operators.h"
#include "sycldnn/softmax/params.h"

#include "src/pointwise/queue_pointwise_forward_impl.h"

namespace sycldnn {
namespace softmax {
namespace internal {

template <typename T, typename Index, typename SoftmaxType, typename Backend,
          int VectorWidth>
SNNStatus queue_softmax(typename Backend::template pointer_type<T const>& input,
                        typename Backend::template pointer_type<T>& workspace,
                        typename Backend::template pointer_type<T>& output,
                        SoftmaxParams const& params, Backend& backend) {
  Index n_items = params.batch * params.channels * params.rows * params.cols;
  auto queue = backend.get_queue();
  auto in_mem = backend.get_mem_object(input, n_items);
  auto out_mem = backend.get_mem_object(output, n_items);
  Index workspace_items = params.batch * params.rows * params.cols;
  auto workspace_mem = backend.get_mem_object(workspace, workspace_items);

  SNNStatus status =
      pointwise::internal::queue_pointwise<T, Index, pointwise::Exp,
                                           pointwise::Forward, VectorWidth>(
          in_mem, out_mem, n_items, queue);

  auto const_output = backend.to_const_pointer(&output);
  backend.template reduce<T, Index, sycldnn::softmax::SoftmaxParams>(
      const_output, workspace, params);

  sycldnn::MemObject<T const, cl::sycl::detail::aligned_mem::aligned_allocator>
      const_workspace_mem = workspace_mem;

  status =
      pointwise::internal::queue_pointwise<T, Index, pointwise::SoftMaxDiv,
                                           pointwise::Forward, VectorWidth>(
          const_workspace_mem, out_mem, n_items, queue);
  return status;
}

}  // namespace internal
}  // namespace softmax
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_SOFTMAX_QUEUE_IMPL_H
