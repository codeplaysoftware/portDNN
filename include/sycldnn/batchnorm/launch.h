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
#ifndef SYCLDNN_INCLUDE_BATCHNORM_LAUNCH_H_
#define SYCLDNN_INCLUDE_BATCHNORM_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::batchnorm::launch() function, which
 * asynchronously dispatches a SYCL kernel to compute a batchnorm operation
 * along a single dimension of a N-dimensional tensor.
 */
#include "sycldnn/status.h"

#include "sycldnn/batchnorm/params.h"

#include "sycldnn/internal/batchnorm/launch_internal.h"

#include "sycldnn/helpers/macros.h"

#include "sycldnn/internal/transpose/launch.h"

namespace sycldnn {
/** Namespace containing the batchnorm operator. */
namespace batchnorm {

/**
 * Compute workspace size required by Batchnorm Forward Training and Batchnorm
 * gradient.
 *
 * \tparam Direction The direction of processing, either Forward or Gradient.
 * \param params The parameters of the batch normalization operation.
 * \return  Returns a SNNStatus containing a StatusCode enum showing
 * if the launch was OK or whether it encountered some problem.
 */
template <typename Direction>
SNNStatus compute_workspace_size(BatchNormParams& params) {
  params.required_workspace_size = 0;

  if constexpr (internal::IsForwardTraining<Direction>) {
    const std::vector<int> lhs_dims = internal::get_input_dims(params);
    const std::vector<int> rhs_dims = internal::get_binaryop_rhs_dims(params);
    std::vector<int> binaryop_out_dims;
    auto status = binaryop::internal::compute_out_dims(lhs_dims, rhs_dims,
                                                       binaryop_out_dims);

    if (status.status != sycldnn::StatusCode::OK) {
      return status;
    }

    params.required_workspace_size =
        std::accumulate(binaryop_out_dims.begin(), binaryop_out_dims.end(), 1,
                        std::multiplies<int>()) +
        params.channels;
  }

  if constexpr (internal::IsGradient<Direction>) {
    params.required_workspace_size =
        params.batch * params.rows * params.cols * params.channels;
  }

  return SNNStatus{{}, sycldnn::StatusCode::OK};
}

/** Namespace containing internal implementation details for batchnorm. */
namespace internal {

/**
 * Validate that the user-provided batchnorm parameters are consistent with what
 * is expected by SYCL-DNN.
 *
 * If compiled with asserts, any invalid parameter will fail with an assert.
 * Otherwise a status code \ref StatusCode::InvalidParameter will be returned.
 *
 * \tparam Direction The direction of processing, either Forward or Gradient.
 * \param params  BatchNorm parameters to validate.
 * \return        A SNNStatus object containing either \ref StatusCode::OK if
 * all parameters are valid, or \ref StatusCode::InvalidParameter otherwise.
 */
template <typename Direction>
SNN_EXPORT SNNStatus inline validate_params(BatchNormParams const& params) {
  SNN_VALIDATE_PARAM(params.batch > 0, "The batch size must be positive.");
  SNN_VALIDATE_PARAM(params.channels > 0,
                     "The number of channels/classes must be positive.");
  SNN_VALIDATE_PARAM(params.rows > 0,
                     "The number of input/output rows must be positive.");
  SNN_VALIDATE_PARAM(params.cols > 0,
                     "The number of input/output columns must be positive.");
  SNN_VALIDATE_PARAM(params.epsilon > 0.f,
                     "The epsilon parameter must be greater than 0.");
  SNN_VALIDATE_PARAM(
      params.momentum >= 0.f,
      "The momentum parameter must be greater than or equal to 0.");

  if constexpr (internal::IsForwardTraining<Direction> ||
                internal::IsGradient<Direction>) {
    BatchNormParams validation_params = params;
    compute_workspace_size<Direction>(validation_params);
    SNN_VALIDATE_PARAM(params.required_workspace_size ==
                           validation_params.required_workspace_size,
                       "The value of params.required_workspace_size must be "
                       "set to the correct value obtained by calling "
                       "sycldnn::batchnorm::compute_workspace_size().");
  }

  return StatusCode::OK;
}

}  // namespace internal

/**
 * Launch BatchNorm forward propagation in Inference mode.
 *
 * BatchNorm is applied along the channel dimension of a 4D tensor - for 2D
 * matrices with shape (batch x channels), the height and width dimensions can
 * be set to 1.
 *
 * For inputs with height and width > 1, batchnorm is applied pixel-wise. This
 * is identical to multiplying the batch-size by the total number of pixels for
 * performing batchnorm on (i.e. batch' = batch x height x width), yielding a 2D
 * matrix as above with dimensions (batch' x channels).
 *
 * In inference mode, BatchNorm uses the provided mean and variance to normalize
 * the input.
 *
 * \tparam T The data type of the input tensor.
 * \tparam Backend The type of backend.
 * \tparam Direction The direction of processing, either Forward or Gradient.
 * \param input A pointer to memory representing the input tensor.
 * \param beta A pointer to memory representing the bias tensor.
 * \param gamma A pointer to memory representing the scale tensor.
 * \param mean A pointer to memory for input mean tensor.
 * \param variance A pointer to memory for input variance tensor.
 * \param events     Events which should be completed before the operation
 * \param output A pointer to memory representing the output tensor.
 * \param params The batchnorm parameters.
 * \param backend The backend for mapping between pointer representations.
 * \return Returns a SNNStatus containing the SYCL event tied to the kernel
 *         launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, typename Backend, typename Direction,
          typename = internal::EnableIfForwardInference<Direction>>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T const> beta,
                 typename Backend::template pointer_type<T const> gamma,
                 typename Backend::template pointer_type<T const> mean,
                 typename Backend::template pointer_type<T const> variance,
                 typename Backend::template pointer_type<T> output,
                 BatchNormParams const& params, Backend& backend,
                 const std::vector<cl::sycl::event>& events = {}) {
  auto validation_status = internal::validate_params<Direction>(params);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }

  auto n_items = params.batch * params.channels * params.rows * params.cols;
  auto input_mem = backend.get_mem_object(input, n_items);
  auto gamma_mem = backend.get_mem_object(gamma, params.channels);
  auto beta_mem = backend.get_mem_object(beta, params.channels);
  auto mean_mem = backend.get_mem_object(mean, params.channels);
  auto var_mem = backend.get_mem_object(variance, params.channels);
  auto output_mem = backend.get_mem_object(output, n_items);

  return internal::launch_inference_internal<T, Backend>(
      input_mem, beta_mem, gamma_mem, mean_mem, var_mem, output_mem, params,
      backend, events);
}

/**
 * Launch BatchNorm forward propagation in Training mode.
 *
 * BatchNorm is applied along the channel dimension of a 4D tensor - for 2D
 * matrices with shape (batch x channels), the height and width dimensions can
 * be set to 1.
 *
 * For inputs with height and width > 1, batchnorm is applied pixel-wise. This
 * is identical to multiplying the batch-size by the total number of pixels for
 * performing batchnorm on (i.e. batch' = batch x height x width), yielding a 2D
 * matrix as above with dimensions (batch' x channels).
 *
 * In training mode, BatchNorm computes the mean and variance of the input
 * tensor, uses them to normalize it, and updates the running statistics based
 * on the specified momentum.
 *
 * \tparam T The data type of the input tensor.
 * \tparam Backend The type of backend.
 * \tparam Direction The direction of processing, either Forward or Gradient.
 * \param input A pointer to memory representing the input tensor.
 * \param beta A pointer to memory representing the bias tensor.
 * \param gamma A pointer to memory representing the scale tensor.
 * \param running_mean A pointer to memory for running mean tensor.
 * \param running_variance A pointer to memory for running variance tensor.
 * \param cache_mean A pointer to memory for running cache mean tensor, used
 *        in the gradient.
 * \param cache_invvariance A pointer to memory for cache inverse variance
 *        tensor used in the gradient.
 * \param events     Events which should be completed before the operation
 * \param output A pointer to memory representing the output tensor.
 * \param workspace A pointer to memory representing the workspace tensor.
 * \param params The batchnorm parameters.
 * \param backend The backend for mapping between pointer representations.
 * \return Returns a SNNStatus containing the SYCL event tied to the kernel
 *         launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, typename Backend, typename Direction,
          typename = internal::EnableIfForwardTraining<Direction>>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T const> beta,
                 typename Backend::template pointer_type<T const> gamma,
                 typename Backend::template pointer_type<T> running_mean,
                 typename Backend::template pointer_type<T> running_variance,
                 typename Backend::template pointer_type<T> cache_mean,
                 typename Backend::template pointer_type<T> cache_invvariance,
                 typename Backend::template pointer_type<T> workspace,
                 typename Backend::template pointer_type<T> output,
                 BatchNormParams const& params, Backend& backend,
                 const std::vector<cl::sycl::event>& events = {}) {
  auto validation_status = internal::validate_params<Direction>(params);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }

  auto n_items = params.batch * params.channels * params.rows * params.cols;
  auto input_mem = backend.get_mem_object(input, n_items);
  auto gamma_mem = backend.get_mem_object(gamma, params.channels);
  auto beta_mem = backend.get_mem_object(beta, params.channels);
  auto running_mean_mem = backend.get_mem_object(running_mean, params.channels);
  auto running_variance_mem =
      backend.get_mem_object(running_variance, params.channels);
  auto cache_mean_mem = backend.get_mem_object(cache_mean, params.channels);
  auto cache_invvariance_mem =
      backend.get_mem_object(cache_invvariance, params.channels);
  auto output_mem = backend.get_mem_object(output, n_items);
  auto workspace_mem =
      backend.get_mem_object(workspace, params.required_workspace_size);
  return internal::launch_training<T, Backend>(
      input_mem, beta_mem, gamma_mem, running_mean_mem, running_variance_mem,
      cache_mean_mem, cache_invvariance_mem, workspace_mem, output_mem, params,
      backend, events);
}

/**
 * \cond Doxygen_Suppress
 * Launch BatchNorm gradient in Training mode.
 *
 * BatchNorm is applied along the channel dimension of a 4D tensor - for 2D
 * matrices with shape (batch x channels), the height and width dimensions can
 * be set to 1.
 *
 * For inputs with height and width > 1, batchnorm is applied pixel-wise. This
 * is identical to multiplying the batch-size by the total number of pixels for
 * performing batchnorm on (i.e. batch' = batch x height x width), yielding a 2D
 * matrix as above with dimensions (batch' x channels).
 *
 * BatchNorm gradient computes the gradient of the input, gamma (scale) and beta
 * (bias) tensors.
 *
 * \tparam T The data type of the input tensor.
 * \tparam Backend The type of backend.
 * \tparam Direction The direction of processing, either Forward or Gradient.
 * \param input A pointer to memory representing the input tensor.
 * \param out_grad A pointer to memory representing the output gradient tensor.
 * \param in_grad A pointer to memory representing the input gradient tensor.
 * \param gamma A pointer to memory representing the scale tensor.
 * \param beta_grad A pointer to memory representing the beta gradient tensor.
 * \param gamma_grad A pointer to memory representing the gamma gradient tensor.
 * \param cache_mean A pointer to memory for the cache mean tensor computed by
          Batchnorm Forward Training.
 * \param cache_invvariance A pointer to memory for the cache inverse variance
          tensor computed by Batchnorm Forward Training.
 * \param workspace A pointer to memory for the workspace tensor.
 * \param params The batchnorm parameters.
 * \param backend The backend for mapping between pointer representations.
 * \param events Events which should be completed before the operation.
 * \return Returns a SNNStatus containing the SYCL event tied to the kernel
 *         launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 * \endcond
 */
template <typename T, typename Backend, typename Direction,
          typename = internal::EnableIfGradient<Direction>>
SNNStatus launch(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T const> out_grad,
    typename Backend::template pointer_type<T> in_grad,
    typename Backend::template pointer_type<T const> gamma,
    typename Backend::template pointer_type<T> beta_grad,
    typename Backend::template pointer_type<T> gamma_grad,
    typename Backend::template pointer_type<T const> cache_mean,
    typename Backend::template pointer_type<T const> cache_invvariance,
    typename Backend::template pointer_type<T> workspace,
    BatchNormParams const& params, Backend& backend,
    const std::vector<cl::sycl::event>& events = {}) {
  auto validation_status = internal::validate_params<Direction>(params);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }

  auto n_items = params.batch * params.channels * params.rows * params.cols;
  auto input_mem = backend.get_mem_object(input, n_items);
  auto out_grad_mem = backend.get_mem_object(out_grad, n_items);
  auto in_grad_mem = backend.get_mem_object(in_grad, n_items);
  auto gamma_mem = backend.get_mem_object(gamma, params.channels);
  auto beta_grad_mem = backend.get_mem_object(beta_grad, params.channels);
  auto gamma_grad_mem = backend.get_mem_object(gamma_grad, params.channels);
  auto cache_mean_mem = backend.get_mem_object(cache_mean, params.channels);
  auto cache_invvar_mem =
      backend.get_mem_object(cache_invvariance, params.channels);

  auto workspace_mem =
      backend.get_mem_object(workspace, params.required_workspace_size);
  return internal::launch_gradient<T, Backend>(
      input_mem, out_grad_mem, in_grad_mem, gamma_mem, beta_grad_mem,
      gamma_grad_mem, cache_mean_mem, cache_invvar_mem, workspace_mem, params,
      backend, events);
}

}  // namespace batchnorm
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_BATCHNORM_LAUNCH_H_
