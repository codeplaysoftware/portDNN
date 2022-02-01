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

#include "sycldnn/binaryop/operators.h"
#include "sycldnn/internal/binaryop/launch.h"

namespace sycldnn {
namespace softmax {
namespace internal {

template <typename Direction>
using EnableIfGradient = typename std::enable_if<
    std::is_same<Direction, sycldnn::softmax::Gradient>::value, int>::type;

template <typename Direction>
using DisableIfGradient = typename std::enable_if<
    !std::is_same<Direction, sycldnn::softmax::Gradient>::value, int>::type;

/**
 * The internal softmax launcher for Forward direction.
 *
 * Performs an element-wise exponentiation, followed by reduction
 * and then the pointwise division.
 */
template <typename T, typename Direction, typename Backend,
          typename = DisableIfGradient<Direction>>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T> workspace,
                 typename Backend::template pointer_type<T> output,
                 SoftmaxParams const& params, Backend& backend) {
  auto n_items = params.batch * params.channels * params.rows * params.cols;
  auto queue = backend.get_queue();
  auto in_mem = backend.get_mem_object(input, n_items);
  auto out_mem = backend.get_mem_object(output, n_items);
  auto workspace_items = params.batch * params.rows * params.cols;

  SNNStatus status = pointwise::internal::launch_pointwise<T, pointwise::Exp,
                                                           pointwise::Forward>(
      in_mem, out_mem, n_items, queue);

  if (sycldnn::StatusCode::OK != status.status) return status;

  using ConstPointer = typename Backend::template pointer_type<T const>;
  using Index = int32_t;
  status.event =
      backend.template reduce_inner<T, Index, softmax::SoftmaxParams,
                                    sycldnn::backend::reduction::Add>(
          ConstPointer{output}, workspace, params);

  auto const_workspace = ConstPointer{workspace};
  auto const_workspace_mem =
      backend.get_mem_object(const_workspace, workspace_items);

  auto const_output = ConstPointer{output};
  auto const_output_mem = backend.get_mem_object(const_output, n_items);

  status =
      binaryop::internal::launch_binaryop<T, binaryop::internal::SoftmaxDiv>(
          const_output_mem, const_workspace_mem, out_mem, params.channels,
          queue);
  return status;
}

/**
 * The internal softmax launcher for Gradient (Backward) direction.
 *
 * Performs an binary elementwise multiplication, followed by summation,
 * subtraction and then again multiplication.
 */
template <typename T, typename Direction, typename Backend,
          typename = EnableIfGradient<Direction>>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T const> gradient,
                 typename Backend::template pointer_type<T> workspace,
                 typename Backend::template pointer_type<T> output,
                 SoftmaxParams const& params, Backend& backend) {
  auto n_items1 = params.batch * params.rows * params.cols * params.channels;
  auto n_items2 = params.batch * params.rows * params.cols;

  auto in_mem = backend.get_mem_object(input, n_items1);
  auto grad_mem = backend.get_mem_object(gradient, n_items1);
  auto workspace_mem = backend.get_mem_object(workspace, n_items1);
  auto out_mem = backend.get_mem_object(output, n_items1);

  auto queue = backend.get_queue();

  SNNStatus status = binaryop::internal::launch_binaryop<T, binaryop::Mul>(
      grad_mem, in_mem, workspace_mem, n_items1, queue);

  if (sycldnn::StatusCode::OK != status.status) return status;

  using ConstPointer = typename Backend::template pointer_type<T const>;
  using Index = int32_t;

  auto const_workspace = ConstPointer{workspace};
  auto const_workspace_mem = backend.get_mem_object(const_workspace, n_items1);

  status.event =
      backend.template reduce_inner<T, Index, softmax::SoftmaxParams,
                                    sycldnn::backend::reduction::Add>(
          const_workspace, output, params);

  auto const_output = ConstPointer{output};
  auto const_output_mem = backend.get_mem_object(const_output, n_items2);

  status =
      binaryop::internal::launch_binaryop<T, binaryop::internal::SoftmaxSub>(
          grad_mem, const_output_mem, workspace_mem, params.channels, queue);

  if (sycldnn::StatusCode::OK != status.status) return status;

  status = binaryop::internal::launch_binaryop<T, binaryop::Mul>(
      const_workspace_mem, in_mem, out_mem, n_items1, queue);

  return status;
}

}  // namespace internal
}  // namespace softmax
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_INTERNAL_SOFTMAX_LAUNCH_INTERNAL_H_
