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

#ifndef PORTDNN_INCLUDE_INTERNAL_SOFTMAX_LAUNCH_INTERNAL_H_
#define PORTDNN_INCLUDE_INTERNAL_SOFTMAX_LAUNCH_INTERNAL_H_

#include "portdnn/status.h"

#include "portdnn/internal/pointwise/launch_internal.h"
#include "portdnn/pointwise/direction.h"
#include "portdnn/pointwise/operators.h"

#include "portdnn/binaryop/operators.h"
#include "portdnn/internal/binaryop/launch.h"

#include "portdnn/internal/reduce/launch.h"
#include "portdnn/reduce/operators.h"

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
 * \copydoc launch<T, sycldnn::softmax::Forward, Backend>()
 * Special case for the Forward Direction and NHWC layout.
 */
template <typename T, typename Backend>
SNNStatus launch_forward_nhwc(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T> workspace,
    typename Backend::template pointer_type<T> output,
    SoftmaxParams const& params, Backend& backend,
    const std::vector<cl::sycl::event>& events) {
  SNN_VALIDATE_PARAM(params.input_format == sycldnn::DataFormat::NHWC,
                     "Unexpected layout");
  auto n_items = params.batch * params.rows * params.cols * params.channels;
  auto queue = backend.get_queue();
  auto in_mem = backend.get_mem_object(input, n_items);
  auto out_mem = backend.get_mem_object(output, n_items);
  auto workspace_items = params.batch * params.rows * params.cols;

  std::vector<cl::sycl::event> dependencies = events;

  using ConstPointer = typename Backend::template pointer_type<T const>;
  SNNStatus status;
  status = reduce::internal::sublaunch<T, reduce::Max>(
      input, workspace, params.batch * params.rows * params.cols,
      params.channels, 1, backend, events);

  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }
  dependencies = std::vector<cl::sycl::event>{status.event};

  auto const_workspace = ConstPointer{workspace};
  auto const_workspace_mem =
      backend.get_mem_object(const_workspace, workspace_items);

  status = binaryop::internal::launch_binaryop<binaryop::Sub>(
      in_mem, const_workspace_mem, out_mem,
      {params.batch, params.rows, params.cols, params.channels},
      {params.batch, params.rows, params.cols, 1}, queue, dependencies);

  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }
  dependencies = std::vector<cl::sycl::event>{status.event};

  auto const_output = ConstPointer{output};
  auto const_output_mem = backend.get_mem_object(const_output, n_items);
  status = pointwise::internal::launch_pointwise<pointwise::Exp>(
      const_output_mem, out_mem, n_items, queue, dependencies);

  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }
  dependencies = std::vector<cl::sycl::event>{status.event};

  status = reduce::internal::sublaunch<T, reduce::Add>(
      ConstPointer{output}, workspace, params.batch * params.rows * params.cols,
      params.channels, 1, backend, dependencies);

  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }
  dependencies = std::vector<cl::sycl::event>{status.event};

  status = binaryop::internal::launch_binaryop<binaryop::Div>(
      const_output_mem, const_workspace_mem, out_mem,
      {params.batch, params.rows, params.cols, params.channels},
      {params.batch, params.rows, params.cols, 1}, queue, dependencies);
  return status;
}

/**
 * \copydoc launch<T, sycldnn::softmax::Forward, Backend>()
 * Special case for the Forward Direction and NCHW layout.
 */
template <typename T, typename Backend>
SNNStatus launch_forward_nchw(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T> workspace,
    typename Backend::template pointer_type<T> output,
    SoftmaxParams const& params, Backend& backend,
    const std::vector<cl::sycl::event>& events) {
  SNN_VALIDATE_PARAM(params.input_format == sycldnn::DataFormat::NCHW,
                     "Unexpected layout");
  auto n_items = params.batch * params.channels * params.rows * params.cols;
  auto queue = backend.get_queue();
  auto in_mem = backend.get_mem_object(input, n_items);
  auto out_mem = backend.get_mem_object(output, n_items);
  auto workspace_items = params.batch * params.rows * params.cols;
  std::vector<cl::sycl::event> dependencies = events;

  using ConstPointer = typename Backend::template pointer_type<T const>;
  SNNStatus status;
  status = reduce::internal::sublaunch<T, reduce::Max>(
      input, workspace, params.batch, params.channels,
      params.rows * params.cols, backend, dependencies);

  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }
  dependencies = std::vector<cl::sycl::event>{status.event};

  auto const_workspace = ConstPointer{workspace};
  auto const_workspace_mem =
      backend.get_mem_object(const_workspace, workspace_items);

  status = binaryop::internal::launch_binaryop<binaryop::Sub>(
      in_mem, const_workspace_mem, out_mem,
      {params.batch, params.channels, params.rows, params.cols},
      {params.batch, 1, params.rows, params.cols}, queue, dependencies);

  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }
  dependencies = std::vector<cl::sycl::event>{status.event};

  auto const_output = ConstPointer{output};
  auto const_output_mem = backend.get_mem_object(const_output, n_items);
  status = pointwise::internal::launch_pointwise<pointwise::Exp>(
      const_output_mem, out_mem, n_items, queue, dependencies);

  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }
  dependencies = std::vector<cl::sycl::event>{status.event};

  status = reduce::internal::sublaunch<T, reduce::Add>(
      ConstPointer{output}, workspace, params.batch, params.channels,
      params.rows * params.cols, backend, dependencies);

  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }
  dependencies = std::vector<cl::sycl::event>{status.event};

  status = binaryop::internal::launch_binaryop<binaryop::Div>(
      const_output_mem, const_workspace_mem, out_mem,
      {params.batch, params.channels, params.rows, params.cols},
      {params.batch, 1, params.rows, params.cols}, queue, dependencies);
  return status;
}

/**
 * The internal softmax launcher for Forward direction.
 *
 * Performs an element-wise exponentiation, followed by reduction
 * and then the pointwise division.
 * The input is subtracted from its maximum value (on the channel dimension)
 * to avoid values overflowing with the exponential. This has no effect on the
 * output.
 */
template <typename T, typename Direction, typename Backend,
          typename = DisableIfGradient<Direction>>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T> workspace,
                 typename Backend::template pointer_type<T> output,
                 SoftmaxParams const& params, Backend& backend,
                 const std::vector<cl::sycl::event>& events) {
  if (params.input_format == sycldnn::DataFormat::NHWC) {
    return launch_forward_nhwc<T, Backend>(input, workspace, output, params,
                                           backend, events);
  } else if (params.input_format == sycldnn::DataFormat::NCHW) {
    return launch_forward_nchw<T, Backend>(input, workspace, output, params,
                                           backend, events);
  }
  SNN_ASSERT(false, "Unsupported layout");
  return SNNStatus(StatusCode::InvalidParameter);
}

/**
 * \copydoc launch<T, sycldnn::softmax::Gradient, Backend>()
 * Special case for the Gradient Direction and NHWC layout.
 */
template <typename T, typename Backend>
SNNStatus launch_gradient_nhwc(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T const> gradient,
    typename Backend::template pointer_type<T> workspace,
    typename Backend::template pointer_type<T> output,
    SoftmaxParams const& params, Backend& backend,
    const std::vector<cl::sycl::event>& events) {
  SNN_VALIDATE_PARAM(params.input_format == sycldnn::DataFormat::NHWC,
                     "Unexpected layout");
  auto n_items1 = params.batch * params.rows * params.cols * params.channels;
  auto n_items2 = params.batch * params.rows * params.cols;

  auto in_mem = backend.get_mem_object(input, n_items1);
  auto grad_mem = backend.get_mem_object(gradient, n_items1);
  auto workspace_mem = backend.get_mem_object(workspace, n_items1);
  auto out_mem = backend.get_mem_object(output, n_items1);
  std::vector<cl::sycl::event> dependencies = events;

  auto queue = backend.get_queue();

  SNNStatus status = binaryop::internal::launch_binaryop<binaryop::Mul>(
      grad_mem, in_mem, workspace_mem, n_items1, queue, dependencies);

  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }
  dependencies = std::vector<cl::sycl::event>{status.event};

  using ConstPointer = typename Backend::template pointer_type<T const>;

  auto const_workspace = ConstPointer{workspace};
  auto const_workspace_mem = backend.get_mem_object(const_workspace, n_items1);

  status = reduce::internal::sublaunch<T, reduce::Add>(
      const_workspace, output, params.batch * params.rows * params.cols,
      params.channels, 1, backend, dependencies);

  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }
  dependencies = std::vector<cl::sycl::event>{status.event};

  auto const_output = ConstPointer{output};
  auto const_output_mem = backend.get_mem_object(const_output, n_items2);

  status = binaryop::internal::launch_binaryop<binaryop::Sub>(
      grad_mem, const_output_mem, workspace_mem,
      {params.batch, params.rows, params.cols, params.channels},
      {params.batch, params.rows, params.cols, 1}, queue, dependencies);

  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }
  dependencies = std::vector<cl::sycl::event>{status.event};

  status = binaryop::internal::launch_binaryop<binaryop::Mul>(
      const_workspace_mem, in_mem, out_mem, n_items1, queue, dependencies);

  return status;
}

/**
 * \copydoc launch<T, sycldnn::softmax::Gradient, Backend>()
 * Special case for the Gradient Direction and NCHW layout.
 */
template <typename T, typename Backend>
SNNStatus launch_gradient_nchw(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T const> gradient,
    typename Backend::template pointer_type<T> workspace,
    typename Backend::template pointer_type<T> output,
    SoftmaxParams const& params, Backend& backend,
    const std::vector<cl::sycl::event>& events) {
  SNN_VALIDATE_PARAM(params.input_format == sycldnn::DataFormat::NCHW,
                     "Unexpected layout");
  auto n_items1 = params.batch * params.channels * params.rows * params.cols;
  auto n_items2 = params.batch * params.rows * params.cols;

  auto in_mem = backend.get_mem_object(input, n_items1);
  auto grad_mem = backend.get_mem_object(gradient, n_items1);
  auto workspace_mem = backend.get_mem_object(workspace, n_items1);
  auto out_mem = backend.get_mem_object(output, n_items1);
  std::vector<cl::sycl::event> dependencies = events;

  auto queue = backend.get_queue();

  SNNStatus status = binaryop::internal::launch_binaryop<binaryop::Mul>(
      grad_mem, in_mem, workspace_mem, n_items1, queue, dependencies);

  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }
  dependencies = std::vector<cl::sycl::event>{status.event};

  using ConstPointer = typename Backend::template pointer_type<T const>;

  auto const_workspace = ConstPointer{workspace};
  auto const_workspace_mem = backend.get_mem_object(const_workspace, n_items1);

  status = reduce::internal::sublaunch<T, reduce::Add>(
      const_workspace, output, params.batch, params.channels,
      params.rows * params.cols, backend, dependencies);

  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }
  dependencies = std::vector<cl::sycl::event>{status.event};

  auto const_output = ConstPointer{output};
  auto const_output_mem = backend.get_mem_object(const_output, n_items2);

  status = binaryop::internal::launch_binaryop<binaryop::Sub>(
      grad_mem, const_output_mem, workspace_mem,
      {params.batch, params.channels, params.rows, params.cols},
      {params.batch, 1, params.rows, params.cols}, queue, dependencies);

  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }
  dependencies = std::vector<cl::sycl::event>{status.event};

  status = binaryop::internal::launch_binaryop<binaryop::Mul>(
      const_workspace_mem, in_mem, out_mem, n_items1, queue, dependencies);

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
                 SoftmaxParams const& params, Backend& backend,
                 const std::vector<cl::sycl::event>& events) {
  if (params.input_format == sycldnn::DataFormat::NHWC) {
    return launch_gradient_nhwc<T, Backend>(input, gradient, workspace, output,
                                            params, backend, events);
  } else if (params.input_format == sycldnn::DataFormat::NCHW) {
    return launch_gradient_nchw<T, Backend>(input, gradient, workspace, output,
                                            params, backend, events);
  }
  SNN_ASSERT(false, "Unsupported layout");
  return SNNStatus(StatusCode::InvalidParameter);
}

}  // namespace internal
}  // namespace softmax
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_INTERNAL_SOFTMAX_LAUNCH_INTERNAL_H_
