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

#ifndef SYCLDNN_INCLUDE_INTERNAL_BATCHNORM_LAUNCH_INTERNAL_H_
#define SYCLDNN_INCLUDE_INTERNAL_BATCHNORM_LAUNCH_INTERNAL_H_

#include "sycldnn/mem_object.h"

#include "sycldnn/export.h"
#include "sycldnn/helpers/dims.h"
#include "sycldnn/status.h"

#include "sycldnn/batchnorm/direction.h"

#include "sycldnn/binaryop/operators.h"
#include "sycldnn/internal/binaryop/launch.h"

#include "sycldnn/internal/pointwise/launch_internal.h"
#include "sycldnn/pointwise/operators.h"

#include "sycldnn/internal/reduce/launch.h"
#include "sycldnn/reduce/operators.h"

#include "sycldnn/helpers/event_handling.h"
#include "sycldnn/helpers/mem_utils.h"
#include "sycldnn/internal/transpose/launch.h"

namespace sycldnn {
namespace batchnorm {
namespace internal {

template <typename Direction>
static constexpr bool IsGradient = std::is_same<Direction, Gradient>::value;

template <typename Direction>
using EnableIfGradient = typename std::enable_if<IsGradient<Direction>>::type;

template <typename Direction>
using DisableIfGradient = typename std::enable_if<!IsGradient<Direction>>::type;

inline std::vector<int> get_input_dims(BatchNormParams const& params) {
  switch (params.input_format) {
    case DataFormat::NHWC:
      return {params.batch, params.rows, params.cols, params.channels};

    case DataFormat::NCHW:
      return {params.batch, params.channels, params.rows, params.cols};
  }
  SNN_ASSERT(false, "Unsupported Layout");
  return {};
}

inline std::vector<int> get_4d_channel_dims(BatchNormParams const& params) {
  switch (params.input_format) {
    case DataFormat::NHWC:
      return {1, 1, 1, params.channels};

    case DataFormat::NCHW:
      return {1, params.channels, 1, 1};
  }
  SNN_ASSERT(false, "Unsupported Layout");
  return {};
}

inline int get_total_size(BatchNormParams const& params) {
  return params.batch * params.rows * params.cols * params.channels;
}

inline int get_non_channel_size(BatchNormParams const& params) {
  return params.batch * params.rows * params.cols;
}

/**
 * The internal launcher for computing variance.
 */
template <typename T, typename Backend, template <typename> class MemObj,
          typename = std::enable_if<is_mem_obj_v<MemObj<T>, T>>>
SNNStatus launch_variance(MemObj<T const>& centered_input, MemObj<T>& variance,
                          MemObj<T>& squared_centered_input,
                          BatchNormParams const& params, Backend& backend,
                          const std::vector<cl::sycl::event>& events) {
  auto queue = backend.get_queue();
  SNNStatus status;
  status = binaryop::internal::launch_binaryop<binaryop::Mul>(
      centered_input, centered_input, squared_centered_input,
      get_total_size(params), queue, events);
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto const_squared_centered_input = squared_centered_input.as_const();
  status = reduce::internal::launch<reduce::Mean>(
      const_squared_centered_input, variance, 1, get_non_channel_size(params),
      params.channels, backend, std::vector<cl::sycl::event>{status.event});
  return status;
}

/**
 * The internal launcher for computing batchnorm.
 */
template <typename T, template <typename> class MemObj,
          typename = std::enable_if<is_mem_obj_v<MemObj<T>, T>>>
inline SNNStatus launch_batchnorm(
    MemObj<T const>& input, MemObj<T const>& beta, MemObj<T const>& gamma,
    MemObj<T const>& current_mean, MemObj<T const>& current_variance,
    MemObj<T>& output, MemObj<T>& centered_input, MemObj<T>& workspace,
    const float epsilon, const std::vector<int>& input_dims,
    const std::vector<int>& channel_dims, cl::sycl::queue& queue,
    const std::vector<cl::sycl::event>& events) {
  constexpr bool is_usm = is_usm_obj_v<MemObj<T>, T>;
  std::vector<cl::sycl::event> dependencies;

  SNNStatus status = binaryop::internal::launch_binaryop<binaryop::Sub>(
      input, current_mean, centered_input, input_dims, channel_dims, queue,
      events);
  dependencies = {status.event};
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto sycl_epsilon =
      sycldnn::helpers::alloc_and_assign<T, is_usm>(1, &epsilon, queue);
  auto epsilon_mem = make_mem_object<T const>(sycl_epsilon, 1);
  status = binaryop::internal::launch_binaryop<binaryop::Add>(
      current_variance, epsilon_mem, workspace, channel_dims, {1}, queue,
      events);
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto const_workspace = workspace.as_const();
  status = pointwise::internal::launch_pointwise<pointwise::Sqrt>(
      const_workspace, workspace, helpers::get_total_size(channel_dims), queue,
      {status.event});
  // Output depends on centered_input and workspace
  dependencies.push_back(status.event);
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto const_centered_input = centered_input.as_const();
  status = binaryop::internal::launch_binaryop<binaryop::Div>(
      const_centered_input, const_workspace, output, input_dims, channel_dims,
      queue, dependencies);
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto const_output = output.as_const();
  status = binaryop::internal::launch_binaryop<binaryop::Mul>(
      const_output, gamma, output, input_dims, channel_dims, queue,
      {status.event});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  status = binaryop::internal::launch_binaryop<binaryop::Add>(
      const_output, beta, output, input_dims, channel_dims, queue,
      {status.event});

  status.event =
      sycldnn::helpers::enqueue_free(queue, {status.event}, sycl_epsilon);

  return status;
}

/**
 * Compute running mean and running variance:
 * output = input * momentum + output * (1 - momentum)
 */
template <typename T, template <typename> class MemObj,
          typename = std::enable_if<is_mem_obj_v<MemObj<T>, T>>>
inline SNNStatus launch_running_mean_variance(
    MemObj<T const>& input, MemObj<T const>& momentum,
    MemObj<T const>& one_minus_momentum, MemObj<T>& output,
    MemObj<T>& workspace, int size, cl::sycl::queue& queue,
    const std::vector<cl::sycl::event>& events) {
  auto const_output = output.as_const();
  SNNStatus status = binaryop::internal::launch_binaryop<binaryop::Mul>(
      const_output, one_minus_momentum, output, {size}, {1}, queue, events);
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  SNNStatus status2 = binaryop::internal::launch_binaryop<binaryop::Mul>(
      input, momentum, workspace, {size}, {1}, queue, events);
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto const_workspace = workspace.as_const();
  status = binaryop::internal::launch_binaryop<binaryop::Add>(
      const_output, const_workspace, output, size, queue,
      {status.event, status2.event});
  return status;
}

/**
 * The internal batchnorm launcher for Forward Direction when computing Mean and
 * Variance.
 *
 * Calculates Mean, then Variance and then Batchnorm.
 */

template <typename T, typename Backend, template <typename> class MemObj,
          typename = std::enable_if<is_mem_obj_v<MemObj<T>, T>>>
SNNStatus launch_forward(MemObj<T const>& input, MemObj<T const>& beta,
                         MemObj<T const>& gamma, MemObj<T const>& input_mean,
                         MemObj<T const>& input_variance,
                         MemObj<T>& running_mean, MemObj<T>& running_variance,
                         MemObj<T>& output, BatchNormParams const& params,
                         Backend& backend,
                         const std::vector<cl::sycl::event>& events) {
  constexpr bool is_usm = is_usm_obj_v<MemObj<T>, T>;
  auto n_items = get_total_size(params);
  auto queue = backend.get_queue();
  auto input_dims = get_input_dims(params);
  auto channel_dims = get_4d_channel_dims(params);

  SNNStatus status;
  std::vector<cl::sycl::event> dependencies = events;

  auto sycl_auxiliary_input =
      sycldnn::helpers::alloc<T, is_usm>(n_items, queue);
  auto auxiliary_input =
      make_mem_object(sycl_auxiliary_input, n_items);  // Reused-later

  auto sycl_workspace =
      sycldnn::helpers::alloc<T, is_usm>(params.channels, queue);
  auto workspace = make_mem_object(sycl_workspace, params.channels);

  // auxiliary_input = centered_input
  status = launch_batchnorm(input, beta, gamma, input_mean, input_variance,
                            output, auxiliary_input, workspace, params.epsilon,
                            input_dims, channel_dims, queue, events);
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  // Transpose NCHW input to NHWC to reduce NHW dimensions in one go.
  const bool is_nchw = params.input_format == DataFormat::NCHW;
  if (is_nchw) {
    status = transpose::internal::launch(input, auxiliary_input, input_dims,
                                         NCHW_TO_NHWC, queue, {status.event});
    if (sycldnn::StatusCode::OK != status.status) {
      return status;
    }
  }
  // auxiliary_input = transposed input
  auto const_tr_input = auxiliary_input.as_const();
  auto& nhwc_input = is_nchw ? const_tr_input : input;
  auto nhwc_dims = {params.batch, params.rows, params.cols, params.channels};

  status = reduce::internal::launch<reduce::Mean>(
      nhwc_input, running_mean, 1, get_non_channel_size(params),
      params.channels, backend, {status.event});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  // auxiliary_input = new centered_input
  auto const_running_mean = running_mean.as_const();
  status = binaryop::internal::launch_binaryop<binaryop::Sub>(
      nhwc_input, const_running_mean, auxiliary_input, nhwc_dims,
      {params.channels}, queue, {status.event});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto sycl_momentum =
      sycldnn::helpers::alloc_and_assign<T, is_usm>(1, &params.momentum, queue);
  auto momentum = make_mem_object<T const>(sycl_momentum, 1);
  const T one_minus_momentum_val = 1 - params.momentum;
  auto sycl_one_minus_momentum = sycldnn::helpers::alloc_and_assign<T, is_usm>(
      1, &one_minus_momentum_val, queue);
  auto one_minus_momentum =
      make_mem_object<T const>(sycl_one_minus_momentum, 1);

  status = launch_running_mean_variance(
      input_mean, momentum, one_minus_momentum, running_mean, workspace,
      params.channels, queue, {status.event});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto const_centered_input = auxiliary_input.as_const();
  status = launch_variance(const_centered_input, running_variance,
                           auxiliary_input, params, backend, {status.event});
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  status = launch_running_mean_variance(
      input_variance, momentum, one_minus_momentum, running_variance, workspace,
      params.channels, queue, {status.event});

  status.event = sycldnn::helpers::enqueue_free(
      queue, {status.event}, sycl_auxiliary_input, sycl_workspace,
      sycl_momentum, sycl_one_minus_momentum);
  return status;
}

/**
 * The internal batchnorm launcher for Forward Direction when using the existing
 * Mean and Variance.
 *
 * Calculates Batchnorm by using the input, beta, gamma, mean and variance.
 * provided by the user.
 */

template <typename T, typename Backend, template <typename> class MemObj,
          typename = std::enable_if<is_mem_obj_v<MemObj<T>, T>>>
SNNStatus launch_forward(MemObj<T const>& input, MemObj<T const>& beta,
                         MemObj<T const>& gamma, MemObj<T const>& running_mean,
                         MemObj<T const>& running_variance, MemObj<T>& output,
                         BatchNormParams const& params, Backend& backend,
                         const std::vector<cl::sycl::event>& events) {
  constexpr bool is_usm = is_usm_obj_v<MemObj<T>, T>;
  auto n_items = get_total_size(params);
  auto queue = backend.get_queue();
  auto input_dims = get_input_dims(params);
  auto channel_dims = get_4d_channel_dims(params);

  auto sycl_centered_input = sycldnn::helpers::alloc<T, is_usm>(n_items, queue);
  auto centered_input = make_mem_object(sycl_centered_input, n_items);

  auto sycl_workspace =
      sycldnn::helpers::alloc<T, is_usm>(params.channels, queue);
  auto workspace = make_mem_object(sycl_workspace, params.channels);

  SNNStatus status =
      launch_batchnorm(input, beta, gamma, running_mean, running_variance,
                       output, centered_input, workspace, params.epsilon,
                       input_dims, channel_dims, queue, events);

  status.event = sycldnn::helpers::enqueue_free(
      queue, {status.event}, sycl_centered_input, sycl_workspace);

  return status;
}

/**
 * The internal batchnorm launcher for Gradient Direction when computing Mean
 * and Variance.
 *
 * Calculates Mean, then Variance and then Batchnorm.
 * https://github.com/tensorflow/tensorflow/blob/d916f20e1f1897696a19158ac7f5bd8d83e1b857/tensorflow/python/ops/nn_grad.py#L924
 */

template <typename T, typename Backend, template <typename> class MemObj,
          typename = std::enable_if<is_mem_obj_v<MemObj<T>, T>>>
SNNStatus launch_gradient(MemObj<T const>& input, MemObj<T const>& gradient,
                          MemObj<T const>& gamma, MemObj<T>& beta_grad,
                          MemObj<T>& gamma_grad, MemObj<T>& output,
                          BatchNormParams const& params, Backend& backend,
                          const std::vector<cl::sycl::event>& events) {
  constexpr bool is_usm = is_usm_obj_v<MemObj<T>, T>;
  auto nhwc_dims = {params.batch, params.rows, params.cols, params.channels};
  auto n_items = get_total_size(params);
  auto queue = backend.get_queue();
  const bool is_nchw = params.input_format == DataFormat::NCHW;
  SNNStatus status_1;
  SNNStatus status_2;
  std::vector<cl::sycl::event> dependencies_1 = events;
  std::vector<cl::sycl::event> dependencies_2 = events;
  std::vector<cl::sycl::event> dependencies_3 = events;

  auto sycl_tr_input = sycldnn::helpers::alloc<T, is_usm>(n_items, queue);
  auto tr_input = make_mem_object(sycl_tr_input, n_items);
  auto sycl_tr_gradient = sycldnn::helpers::alloc<T, is_usm>(n_items, queue);
  auto tr_gradient = make_mem_object(sycl_tr_gradient, n_items);
  // Transpose NCHW input and gradient to NHWC to reduce NHW dimensions in one
  // go.
  if (is_nchw) {
    auto input_dims = get_input_dims(params);
    status_1 = transpose::internal::launch(gradient, tr_gradient, input_dims,
                                           NCHW_TO_NHWC, queue, events);
    dependencies_1 = std::vector<cl::sycl::event>{status_1.event};
    dependencies_3 = std::vector<cl::sycl::event>{status_1.event};
    if (sycldnn::StatusCode::OK != status_1.status) {
      return status_1;
    }

    status_2 = transpose::internal::launch(input, tr_input, input_dims,
                                           NCHW_TO_NHWC, queue, events);
    dependencies_2 = std::vector<cl::sycl::event>{status_2.event};
    if (sycldnn::StatusCode::OK != status_2.status) {
      return status_2;
    }
  }

  auto const_tr_gradient = tr_gradient.as_const();
  auto& nhwc_gradient = is_nchw ? const_tr_gradient : gradient;

  status_1 = reduce::internal::launch<reduce::Add>(
      nhwc_gradient, beta_grad, 1, get_non_channel_size(params),
      params.channels, backend, dependencies_1);
  dependencies_1 = std::vector<cl::sycl::event>{status_1.event};
  if (sycldnn::StatusCode::OK != status_1.status) {
    return status_1;
  }

  auto sycl_mean_gradient =
      sycldnn::helpers::alloc<T, is_usm>(params.channels, queue);
  auto mean_gradient = make_mem_object(sycl_mean_gradient, params.channels);
  T num_elts_val = get_non_channel_size(params);
  auto sycl_num_elts =
      sycldnn::helpers::alloc_and_assign<T, is_usm>(1, &num_elts_val, queue);
  auto num_elts = make_mem_object<T const>(sycl_num_elts, 1);
  auto const_beta_grad = beta_grad.as_const();
  status_1 = binaryop::internal::launch_binaryop<binaryop::Div>(
      const_beta_grad, num_elts, mean_gradient, {params.channels}, {1}, queue,
      dependencies_1);
  dependencies_1 = std::vector<cl::sycl::event>{status_1.event};
  if (sycldnn::StatusCode::OK != status_1.status) {
    return status_1;
  }

  auto const_mean_gradient = mean_gradient.as_const();
  status_1 =
      sycldnn::binaryop::internal::launch_binaryop<sycldnn::binaryop::Sub>(
          nhwc_gradient, const_mean_gradient, output, nhwc_dims,
          {params.channels}, queue, dependencies_1);
  dependencies_1 = std::vector<cl::sycl::event>{status_1.event};
  if (sycldnn::StatusCode::OK != status_1.status) {
    return status_1;
  }

  auto const_tr_input = tr_input.as_const();
  auto& nhwc_input = is_nchw ? const_tr_input : input;
  auto sycl_mean_input =
      sycldnn::helpers::alloc<T, is_usm>(params.channels, queue);
  auto mean_input = make_mem_object(sycl_mean_input, params.channels);
  status_1 = reduce::internal::launch<reduce::Mean>(
      nhwc_input, mean_input, 1, get_non_channel_size(params), params.channels,
      backend, dependencies_2);
  dependencies_2 = std::vector<cl::sycl::event>{status_1.event};
  if (sycldnn::StatusCode::OK != status_1.status) {
    return status_1;
  }

  auto sycl_centered_input = sycldnn::helpers::alloc<T, is_usm>(n_items, queue);
  auto centered_input = make_mem_object(sycl_centered_input, n_items);
  auto const_mean_input = mean_input.as_const();
  status_1 =
      sycldnn::binaryop::internal::launch_binaryop<sycldnn::binaryop::Sub>(
          nhwc_input, const_mean_input, centered_input, nhwc_dims,
          {params.channels}, queue, dependencies_2);
  dependencies_3.push_back(status_1.event);
  if (sycldnn::StatusCode::OK != status_1.status) {
    return status_1;
  }

  auto sycl_scaled_input = sycldnn::helpers::alloc<T, is_usm>(n_items, queue);
  auto scaled_input = make_mem_object(sycl_scaled_input, n_items);
  auto const_centered_input = centered_input.as_const();
  status_1 =
      sycldnn::binaryop::internal::launch_binaryop<sycldnn::binaryop::Mul>(
          nhwc_gradient, const_centered_input, scaled_input, nhwc_dims, queue,
          dependencies_3);
  dependencies_2 = std::vector<cl::sycl::event>{status_1.event};
  if (sycldnn::StatusCode::OK != status_1.status) {
    return status_1;
  }

  auto const_scaled_input = scaled_input.as_const();
  status_1 = reduce::internal::launch<reduce::Add>(
      const_scaled_input, gamma_grad, 1, get_non_channel_size(params),
      params.channels, backend, dependencies_2);
  dependencies_2 = std::vector<cl::sycl::event>{status_1.event};
  if (sycldnn::StatusCode::OK != status_1.status) {
    return status_1;
  }

  auto sycl_input_variance =
      sycldnn::helpers::alloc<T, is_usm>(params.channels, queue);
  auto input_variance = make_mem_object(sycl_input_variance, params.channels);
  status_1 = launch_variance(const_centered_input, input_variance, scaled_input,
                             params, backend, dependencies_2);
  dependencies_3 = std::vector<cl::sycl::event>{status_1.event};
  if (sycldnn::StatusCode::OK != status_1.status) {
    return status_1;
  }

  auto sycl_workspace =
      sycldnn::helpers::alloc<T, is_usm>(params.channels, queue);
  auto workspace = make_mem_object(sycl_workspace, params.channels);
  auto const_gamma_grad = gamma_grad.as_const();
  status_1 = binaryop::internal::launch_binaryop<binaryop::Div>(
      const_gamma_grad, num_elts, workspace, {params.channels}, {1}, queue,
      dependencies_2);
  dependencies_2 = std::vector<cl::sycl::event>{status_1.event};
  if (sycldnn::StatusCode::OK != status_1.status) {
    return status_1;
  }

  auto sycl_epsilon =
      sycldnn::helpers::alloc_and_assign<T, is_usm>(1, &params.epsilon, queue);
  auto epsilon = make_mem_object<T const>(sycl_epsilon, 1);
  auto const_input_variance = input_variance.as_const();
  status_1 = binaryop::internal::launch_binaryop<binaryop::Add>(
      const_input_variance, epsilon, input_variance, {params.channels}, {1},
      queue, dependencies_3);
  dependencies_2.push_back(status_1.event);
  if (sycldnn::StatusCode::OK != status_1.status) {
    return status_1;
  }

  auto const_workspace = workspace.as_const();
  status_1 = binaryop::internal::launch_binaryop<binaryop::Div>(
      const_workspace, const_input_variance, workspace, params.channels, queue,
      dependencies_2);
  dependencies_2 = std::vector<cl::sycl::event>{status_1.event};
  if (sycldnn::StatusCode::OK != status_1.status) {
    return status_1;
  }

  status_1 = pointwise::internal::launch_pointwise<pointwise::Sqrt>(
      const_input_variance, input_variance, params.channels, queue,
      dependencies_2);
  dependencies_3 = std::vector<cl::sycl::event>{status_1.event};
  if (sycldnn::StatusCode::OK != status_1.status) {
    return status_1;
  }

  status_1 = binaryop::internal::launch_binaryop<binaryop::Mul>(
      const_centered_input, const_workspace, centered_input, nhwc_dims,
      {params.channels}, queue, dependencies_2);
  dependencies_1.push_back(status_1.event);
  if (sycldnn::StatusCode::OK != status_1.status) {
    return status_1;
  }

  auto const_output = output.as_const();
  status_1 = binaryop::internal::launch_binaryop<binaryop::Sub>(
      const_output, const_centered_input, output, nhwc_dims, queue,
      dependencies_1);
  dependencies_1 = std::vector<cl::sycl::event>{status_1.event};
  if (sycldnn::StatusCode::OK != status_1.status) {
    return status_1;
  }

  status_1 = binaryop::internal::launch_binaryop<binaryop::Mul>(
      const_output, gamma, output, nhwc_dims, {params.channels}, queue,
      dependencies_1);
  dependencies_1 = std::vector<cl::sycl::event>{status_1.event};
  if (sycldnn::StatusCode::OK != status_1.status) {
    return status_1;
  }

  auto& tr_output = tr_input;  // Re-use temporary memory
  status_1 = binaryop::internal::launch_binaryop<binaryop::Div>(
      const_output, const_input_variance, is_nchw ? tr_output : output,
      nhwc_dims, {params.channels}, queue, dependencies_1);
  dependencies_1 = std::vector<cl::sycl::event>{status_1.event};
  if (sycldnn::StatusCode::OK != status_1.status) {
    return status_1;
  }

  // Transpose NHWC output back to NCHW.
  if (is_nchw) {
    auto const_tr_output = tr_output.as_const();
    status_1 = transpose::internal::launch(const_tr_output, output, nhwc_dims,
                                           NHWC_TO_NCHW, queue, dependencies_1);
    if (sycldnn::StatusCode::OK != status_1.status) {
      return status_1;
    }
  }

  status_2 = binaryop::internal::launch_binaryop<binaryop::Div>(
      const_gamma_grad, const_input_variance, gamma_grad, params.channels,
      queue, dependencies_3);

  status_1.event = sycldnn::helpers::enqueue_free(
      queue, std::vector<cl::sycl::event>{status_1.event, status_2.event},
      sycl_tr_input, sycl_tr_gradient, sycl_mean_input, sycl_centered_input,
      sycl_scaled_input, sycl_input_variance, sycl_epsilon, sycl_mean_gradient,
      sycl_num_elts, sycl_workspace);

  return status_1;
}

/**
 * The internal batchnorm launcher for Gradient Direction when using the
 * existing Mean and Variance.
 *
 * Calculates Batchnorm by using the input, beta, gamma, mean and variance.
 * provided by the user.
 */

template <typename T, typename Backend, template <typename> class MemObj,
          typename = std::enable_if<is_mem_obj_v<MemObj<T>, T>>>
SNNStatus launch_gradient(MemObj<T const>& input, MemObj<T const>& gradient,
                          MemObj<T const>& gamma, MemObj<T const>& pop_mean,
                          MemObj<T const>& pop_variance, MemObj<T>& beta_grad,
                          MemObj<T>& gamma_grad, MemObj<T>& output,
                          BatchNormParams const& params, Backend& backend,
                          const std::vector<cl::sycl::event>& events) {
  constexpr bool is_usm = is_usm_obj_v<MemObj<T>, T>;
  auto input_dims = get_input_dims(params);
  auto channel_dims = get_4d_channel_dims(params);
  auto n_items = get_total_size(params);
  auto queue = backend.get_queue();
  const bool is_nchw = params.input_format == DataFormat::NCHW;
  SNNStatus status;

  // Transpose NCHW tensor to NHWC to reduce NHW dimensions in one go.
  auto sycl_tr_reduce = sycldnn::helpers::alloc<T, is_usm>(n_items, queue);
  auto tr_reduce = make_mem_object(sycl_tr_reduce, n_items);
  auto beta_grad_dependencies = std::vector<cl::sycl::event>{};
  if (is_nchw) {
    status = transpose::internal::launch(gradient, tr_reduce, input_dims,
                                         NCHW_TO_NHWC, queue, events);
    beta_grad_dependencies.push_back(status.event);
    if (sycldnn::StatusCode::OK != status.status) {
      return status;
    }
  }
  auto const_tr_reduce = tr_reduce.as_const();
  MemObj<T const>& nhwc_reduce_1 = is_nchw ? const_tr_reduce : gradient;
  SNNStatus beta_grad_status = reduce::internal::launch<reduce::Add>(
      nhwc_reduce_1, beta_grad, 1, get_non_channel_size(params),
      params.channels, backend, beta_grad_dependencies);
  auto launch_gradient_dependencies =
      std::vector<cl::sycl::event>{status.event};
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto sycl_epsilon =
      sycldnn::helpers::alloc_and_assign<T, is_usm>(1, &params.epsilon, queue);
  auto epsilon = make_mem_object<T const>(sycl_epsilon, 1);
  auto sycl_workspace =
      sycldnn::helpers::alloc<T, is_usm>(params.channels, queue);
  auto workspace = make_mem_object(sycl_workspace, params.channels);

  status = binaryop::internal::launch_binaryop<binaryop::Add>(
      pop_variance, epsilon, workspace, channel_dims, {1}, queue, events);
  auto dependencies = std::vector<cl::sycl::event>{status.event};
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto const_workspace = workspace.as_const();
  status = pointwise::internal::launch_pointwise<pointwise::Sqrt>(
      const_workspace, workspace, params.channels, queue, dependencies);
  dependencies = std::vector<cl::sycl::event>{status.event};
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  status = sycldnn::binaryop::internal::launch_binaryop<sycldnn::binaryop::Sub>(
      input, pop_mean, output, input_dims, channel_dims, queue, events);
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  auto const_output = output.as_const();
  status = binaryop::internal::launch_binaryop<binaryop::Mul>(
      const_output, gradient, output, input_dims, queue,
      std::vector<cl::sycl::event>{status.event});
  dependencies.push_back(status.event);
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  status = binaryop::internal::launch_binaryop<binaryop::Div>(
      const_output, const_workspace, output, input_dims, channel_dims, queue,
      dependencies);
  dependencies = std::vector<cl::sycl::event>{status.event};
  auto gamma_grad_dependencies = std::vector<cl::sycl::event>{status.event};
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  // Transpose NCHW tensor to NHWC to reduce NHW dimensions in one go.
  if (is_nchw) {
    status = transpose::internal::launch(
        const_output, tr_reduce, input_dims, NCHW_TO_NHWC, queue,
        std::vector<cl::sycl::event>{status.event, beta_grad_status.event});
    gamma_grad_dependencies = std::vector<cl::sycl::event>{status.event};
    if (sycldnn::StatusCode::OK != status.status) {
      return status;
    }
    const_tr_reduce = tr_reduce.as_const();
  }
  MemObj<T const>& nhwc_reduce_2 = is_nchw ? const_tr_reduce : const_output;

  status = reduce::internal::launch<reduce::Add>(
      nhwc_reduce_2, gamma_grad, 1, get_non_channel_size(params),
      params.channels, backend, gamma_grad_dependencies);
  launch_gradient_dependencies.push_back(status.event);
  dependencies = std::vector<cl::sycl::event>{status.event};
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  // Being dependent on const_output, const_workspace binary op ensures
  // values in workspace have been already used
  status = binaryop::internal::launch_binaryop<binaryop::Div>(
      gamma, const_workspace, workspace, params.channels, queue, dependencies);
  dependencies = std::vector<cl::sycl::event>{status.event};
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  status = binaryop::internal::launch_binaryop<binaryop::Mul>(
      gradient, const_workspace, output, input_dims, channel_dims, queue,
      dependencies);
  dependencies = std::vector<cl::sycl::event>{status.event};
  if (sycldnn::StatusCode::OK != status.status) {
    return status;
  }

  status.event = sycldnn::helpers::enqueue_free(
      queue, std::vector<cl::sycl::event>{status.event}, sycl_epsilon,
      sycl_workspace, sycl_tr_reduce);
  return status;
}

}  // namespace internal
}  // namespace batchnorm
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_INTERNAL_BATCHNORM_LAUNCH_INTERNAL_H_
