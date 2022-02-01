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

#include "sycldnn/status.h"

#include "sycldnn/export.h"

#include "sycldnn/batchnorm/direction.h"
#include "sycldnn/batchnorm/operation.h"
#include "sycldnn/internal/batchnorm/launch_batchnorm.h"

#include "sycldnn/binaryop/operators.h"
#include "sycldnn/internal/binaryop/launch.h"

namespace sycldnn {
namespace batchnorm {
namespace internal {

template <typename Direction, typename Operation>
using EnableIf_Forward_Training = typename std::enable_if<
    std::is_same<Direction, sycldnn::batchnorm::Forward>::value &&
        std::is_same<Operation, sycldnn::batchnorm::Training>::value,
    int>::type;

template <typename Direction, typename Operation>
using DisableIf_Forward_Training = typename std::enable_if<
    std::is_same<Direction, sycldnn::batchnorm::Forward>::value &&
        !std::is_same<Operation, sycldnn::batchnorm::Training>::value,
    int>::type;

template <typename Direction, typename Operation>
using EnableIf_Gradient_Training = typename std::enable_if<
    !std::is_same<Direction, sycldnn::batchnorm::Forward>::value &&
        std::is_same<Operation, sycldnn::batchnorm::Training>::value,
    int>::type;

template <typename Direction, typename Operation>
using DisableIf_Gradient_Training = typename std::enable_if<
    !std::is_same<Direction, sycldnn::batchnorm::Forward>::value &&
        !std::is_same<Operation, sycldnn::batchnorm::Training>::value,
    int>::type;

/**
 * The internal batchnorm launcher for Forward Direction when computing Mean and
 * Variance.
 *
 * Calculates Mean, then Variance and then Batchnorm.
 */

template <typename T, typename Backend, typename Direction, typename Operation,
          typename = EnableIf_Forward_Training<Direction, Operation>>
SNNStatus launch_forward(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T const> beta,
    typename Backend::template pointer_type<T const> gamma,
    typename Backend::template pointer_type<T const> input_mean,
    typename Backend::template pointer_type<T const> input_variance,
    typename Backend::template pointer_type<T> running_mean,
    typename Backend::template pointer_type<T> running_variance,
    typename Backend::template pointer_type<T> output,
    BatchNormParams const& params, Backend& backend) {
  using Index = int32_t;
  SNNStatus status;
  status.status = StatusCode::OK;
  status.event =
      backend
          .template reduce_outer<T, Index, sycldnn::batchnorm::BatchNormParams,
                                 sycldnn::backend::reduction::Mean>(
              input, running_mean, params);

  auto n_items = params.batch * params.channels * params.rows * params.cols;

  auto queue = backend.get_queue();

  using ConstPointer = typename Backend::template pointer_type<T const>;
  auto const_mean = ConstPointer{running_mean};
  auto const_mean_mem = backend.get_mem_object(const_mean, params.channels);

  auto in_mem = backend.get_mem_object(input, n_items);
  auto variance_mem = backend.get_mem_object(running_variance, params.channels);
  auto beta_mem = backend.get_mem_object(beta, params.channels);
  auto gamma_mem = backend.get_mem_object(gamma, params.channels);
  auto out_mem = backend.get_mem_object(output, n_items);

  status = launch_variance(in_mem, const_mean_mem, variance_mem, params, queue);

  if (sycldnn::StatusCode::OK != status.status) return status;

  auto const_variance = ConstPointer{running_variance};
  auto const_variance_mem =
      backend.get_mem_object(const_variance, params.channels);

  status = launch_batchnorm(in_mem, beta_mem, gamma_mem, const_mean_mem,
                            const_variance_mem, out_mem, params, queue);

  if (sycldnn::StatusCode::OK != status.status) return status;

  auto input_mean_mem = backend.get_mem_object(input_mean, params.channels);
  auto running_mean_mem = backend.get_mem_object(running_mean, params.channels);
  auto input_variance_mem =
      backend.get_mem_object(input_variance, params.channels);
  auto running_variance_mem =
      backend.get_mem_object(running_variance, params.channels);

  status = launch_running_mean_variance(
      input_mean_mem, input_variance_mem, running_mean_mem,
      running_variance_mem, params.channels, params.momentum, queue);

  return status;
}

/**
 * The internal batchnorm launcher for Forward Direction when using the existing
 * Mean and Variance.
 *
 * Calculates Batchnorm by using the input, beta, gamma, mean and variance.
 * provided by the user.
 */

template <typename T, typename Direction, typename Operation,
          typename = DisableIf_Forward_Training<Direction, Operation>>
SNNStatus launch_forward(
    BaseMemObject<T const>& input, BaseMemObject<T const>& beta,
    BaseMemObject<T const>& gamma, BaseMemObject<T const>& running_mean,
    BaseMemObject<T const>& running_variance, BaseMemObject<T>& output,
    BatchNormParams const& params, cl::sycl::queue& queue) {
  return launch_batchnorm(input, beta, gamma, running_mean, running_variance,
                          output, params, queue);
}

/**
 * The internal batchnorm launcher for Gradient Direction when computing Mean
 * and Variance.
 *
 * Calculates Mean, then Variance and then Batchnorm.
 * https://github.com/tensorflow/tensorflow/blob/d916f20e1f1897696a19158ac7f5bd8d83e1b857/tensorflow/python/ops/nn_grad.py#L924
 */

template <typename T, typename Backend, typename Direction, typename Operation,
          typename = EnableIf_Gradient_Training<Direction, Operation>>
SNNStatus launch_grad(typename Backend::template pointer_type<T const> input,
                      typename Backend::template pointer_type<T const> gradient,
                      typename Backend::template pointer_type<T const> gamma,
                      typename Backend::template pointer_type<T> workspace,
                      typename Backend::template pointer_type<T> beta_grad,
                      typename Backend::template pointer_type<T> gamma_grad,
                      typename Backend::template pointer_type<T> output,
                      BatchNormParams const& params, Backend& backend) {
  SNNStatus status;
  status.status = StatusCode::OK;
  status.event =
      backend.template reduce_outer<T, int32_t,
                                    sycldnn::batchnorm::BatchNormParams,
                                    sycldnn::backend::reduction::Mean>(
          input, gamma_grad, params);  // mean_x

  auto n_items = params.batch * params.channels * params.rows * params.cols;
  using ConstPointer = typename Backend::template pointer_type<T const>;
  auto queue = backend.get_queue();

  auto const_input_mean = ConstPointer{gamma_grad};
  auto const_input_mean_mem =
      backend.get_mem_object(const_input_mean, params.channels);
  auto input_mem = backend.get_mem_object(input, n_items);
  auto input_variance_mem = backend.get_mem_object(beta_grad, params.channels);

  status = launch_variance(input_mem, const_input_mean_mem, input_variance_mem,
                           params, queue);  // var_x
  if (sycldnn::StatusCode::OK != status.status) return status;

  auto gradient_mem = backend.get_mem_object(gradient, n_items);
  auto workspace_mem = backend.get_mem_object(workspace, n_items);

  status =
      sycldnn::binaryop::internal::launch_binaryop<T, sycldnn::binaryop::Sub>(
          input_mem, const_input_mean_mem, workspace_mem, params.channels,
          queue);  // x_offset

  if (sycldnn::StatusCode::OK != status.status) return status;

  status.event =
      backend.template reduce_outer<T, int32_t,
                                    sycldnn::batchnorm::BatchNormParams,
                                    sycldnn::backend::reduction::Mean>(
          gradient, gamma_grad, params);  // mean_grad_y

  auto const_gradient_mean = ConstPointer{gamma_grad};
  auto const_gradient_mean_mem =
      backend.get_mem_object(const_gradient_mean, params.channels);
  auto output_mem = backend.get_mem_object(output, n_items);

  status =
      sycldnn::binaryop::internal::launch_binaryop<T, sycldnn::binaryop::Sub>(
          gradient_mem, const_gradient_mean_mem, output_mem, params.channels,
          queue);  // grad_y_offset

  if (sycldnn::StatusCode::OK != status.status) return status;

  auto const_workspace = ConstPointer{workspace};
  auto const_workspace_mem = backend.get_mem_object(const_workspace, n_items);

  using Pointer = typename Backend::template pointer_type<T>;

  auto secondary_workspace = Pointer{workspace + static_cast<size_t>(n_items)};
  auto secondary_workspace_mem =
      backend.get_mem_object(secondary_workspace, n_items);

  status =
      sycldnn::binaryop::internal::launch_binaryop<T, sycldnn::binaryop::Mul>(
          gradient_mem, const_workspace_mem, secondary_workspace_mem, n_items,
          queue);  // mean pt 1

  if (sycldnn::StatusCode::OK != status.status) return status;

  status.event =
      backend.template reduce_outer<T, int32_t,
                                    sycldnn::batchnorm::BatchNormParams,
                                    sycldnn::backend::reduction::Mean>(
          secondary_workspace, gamma_grad, params);  // mean pt 2

  auto const_input_variance = ConstPointer{beta_grad};
  auto const_input_variance_mem =
      backend.get_mem_object(const_input_variance, params.channels);

  auto const_mean = ConstPointer{gamma_grad};
  auto const_mean_mem = backend.get_mem_object(const_mean, params.channels);

  auto gamma_mem = backend.get_mem_object(gamma, params.channels);

  status = launch_input_gradient(
      gamma_mem, const_input_variance_mem, const_mean_mem, const_workspace_mem,
      output_mem, params.channels, params.epsilon, queue);  // grad_x

  if (sycldnn::StatusCode::OK != status.status) return status;

  status.event =
      backend.template reduce_outer<T, int32_t,
                                    sycldnn::batchnorm::BatchNormParams,
                                    sycldnn::backend::reduction::Add>(
          secondary_workspace, workspace, params);  // grad_scale pt 1

  if (sycldnn::StatusCode::OK != status.status) return status;

  auto gamma_grad_mem = backend.get_mem_object(gamma_grad, params.channels);

  status = launch_gamma_gradient(const_input_variance_mem, const_workspace_mem,
                                 gamma_grad_mem, params.channels,
                                 params.epsilon, queue);  // grad_scale pt 2

  if (sycldnn::StatusCode::OK != status.status) return status;

  status.event =
      backend.template reduce_outer<T, int32_t,
                                    sycldnn::batchnorm::BatchNormParams,
                                    sycldnn::backend::reduction::Add>(
          gradient, beta_grad, params);  // grad_offset
  return status;
}

/**
 * The internal batchnorm launcher for Gradient Direction when using the
 * existing Mean and Variance.
 *
 * Calculates Batchnorm by using the input, beta, gamma, mean and variance.
 * provided by the user.
 */

template <typename T, typename Backend, typename Direction, typename Operation,
          typename = DisableIf_Gradient_Training<Direction, Operation>>
SNNStatus launch_grad(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T const> gradient,
    typename Backend::template pointer_type<T const> gamma,
    typename Backend::template pointer_type<T const> pop_mean,
    typename Backend::template pointer_type<T const> pop_variance,
    typename Backend::template pointer_type<T> beta_grad,
    typename Backend::template pointer_type<T> gamma_grad,
    typename Backend::template pointer_type<T> output,
    BatchNormParams const& params, Backend& backend) {
  auto n_items = params.batch * params.rows * params.cols * params.channels;
  auto gradient_mem = backend.get_mem_object(gradient, n_items);
  auto input_mem = backend.get_mem_object(input, n_items);
  auto mean_mem = backend.get_mem_object(pop_mean, params.channels);
  auto variance_mem = backend.get_mem_object(pop_variance, params.channels);
  auto output_mem = backend.get_mem_object(output, n_items);

  auto queue = backend.get_queue();
  SNNStatus status = launch_gamma_gradient(
      gradient_mem, input_mem, mean_mem, variance_mem, output_mem,
      params.channels, params.epsilon, queue);  // grad_scale pt 1

  if (sycldnn::StatusCode::OK != status.status) return status;

  status.event =
      backend.template reduce_outer<T, int32_t,
                                    sycldnn::batchnorm::BatchNormParams,
                                    sycldnn::backend::reduction::Add>(
          output, gamma_grad, params);  // grad_scale pt 2

  auto gamma_mem = backend.get_mem_object(gamma, params.channels);

  status =
      launch_input_gradient(gradient_mem, gamma_mem, variance_mem, output_mem,
                            params.channels, params.epsilon, queue);  // grad_x

  if (sycldnn::StatusCode::OK != status.status) return status;

  status.event =
      backend.template reduce_outer<T, int32_t,
                                    sycldnn::batchnorm::BatchNormParams,
                                    sycldnn::backend::reduction::Add>(
          gradient, beta_grad, params);  // grad_offset

  return status;
}

}  // namespace internal
}  // namespace batchnorm
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_INTERNAL_BATCHNORM_LAUNCH_INTERNAL_H_
