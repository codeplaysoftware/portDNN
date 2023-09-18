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
#ifndef PORTDNN_INCLUDE_BATCHNORM_LAUNCH_H_
#define PORTDNN_INCLUDE_BATCHNORM_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::batchnorm::launch() function, which
 * asynchronously dispatches a SYCL kernel to compute a batchnorm operation
 * along a single dimension of a N-dimensional tensor.
 */
#include "portdnn/status.h"

#include "portdnn/batchnorm/params.h"

#include "portdnn/internal/batchnorm/launch_internal.h"

#include "portdnn/helpers/macros.h"

namespace sycldnn {
/** Namespace containing the batchnorm operator. */
namespace batchnorm {
/** Namespace containing internal implementation details for batchnorm. */
namespace internal {

/**
 * Validate that the user-provided batchnorm parameters are consistent with what
 * is expected by portDNN.
 *
 * If compiled with asserts, any invalid parameter will fail with an assert.
 * Otherwise a status code \ref StatusCode::InvalidParameter will be returned.
 *
 * \param params  BatchNorm parameters to validate.
 * \return        A SNNStatus object containing either \ref StatusCode::OK if
 * all parameters are valid, or \ref StatusCode::InvalidParameter otherwise.
 */
SNNStatus inline validate_params(BatchNormParams const& params) {
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
  return StatusCode::OK;
}

/**
 * Generic function to launch batchnorm frozen or training, forward or gradient.
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
 * \tparam T The data type of the input tensor.
 * \tparam Backend The type of backend.
 * \tparam Direction The direction of processing, either Forward or Gradient.
 * \param input A pointer to memory representing the input tensor.
 * \param beta_or_gradient A pointer to memory representing the beta tensor for
 *                         forward batchnorm or the gradient tensor for
 *                         gradient batchnorm.
 * \param gamma A pointer to memory representing the gamma tensor.
 * \param input_mean A pointer to memory for input mean tensor. This is ignored
 *                   for gradient training batchnorm.
 * \param input_variance A pointer to memory for input variance tensor. This is
 *                       ignored for gradient training batchnorm.
 * \param running_mean_or_beta_grad A pointer to memory representing the output
 *                                  mean tensor for forward training batchnorm
 *                                  or the beta_grad tensor for gradient
 *                                  batchnorm. This is ignored for forward
 *                                  frozen batchnorm.
 * \param running_variance_or_gamma_grad A pointer to memory representing the
 *                                       output variance tensor for forward
 *                                       training batchnorm or the gamma_grad
 *                                       tensor for gradient batchnorm. This is
 *                                       ignored for forward frozen batchnorm.
 * \param events     Events which should be completed before the operation
 * \param output A pointer to memory representing the output tensor.
 * \param params The batchnorm parameters.
 * \param backend The backend for mapping between pointer representations.
 * \return Returns a SNNStatus containing the SYCL event tied to the kernel
 *         launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, typename Backend, typename Direction>
SNNStatus sublaunch(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T const> beta_or_gradient,
    typename Backend::template pointer_type<T const> gamma,
    typename Backend::template pointer_type<T const> input_mean,
    typename Backend::template pointer_type<T const> input_variance,
    typename Backend::template pointer_type<T> running_mean_or_beta_grad,
    typename Backend::template pointer_type<T> running_variance_or_gamma_grad,
    typename Backend::template pointer_type<T> output,
    BatchNormParams const& params, Backend& backend,
    const std::vector<cl::sycl::event>& events) {
  auto validation_status = internal::validate_params(params);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }

  auto n_items = params.batch * params.channels * params.rows * params.cols;
  auto input_mem = backend.get_mem_object(input, n_items);
  auto gamma_mem = backend.get_mem_object(gamma, params.channels);
  auto output_mem = backend.get_mem_object(output, n_items);

  if (!internal::IsGradient<Direction>) {
    auto beta_mem = backend.get_mem_object(beta_or_gradient, params.channels);
    auto input_mean_mem = backend.get_mem_object(input_mean, params.channels);
    auto input_variance_mem =
        backend.get_mem_object(input_variance, params.channels);
    if (params.is_training) {
      auto running_mean_mem =
          backend.get_mem_object(running_mean_or_beta_grad, params.channels);
      auto running_variance_mem = backend.get_mem_object(
          running_variance_or_gamma_grad, params.channels);
      // Launch forward training
      return internal::launch_forward<T, Backend>(
          input_mem, beta_mem, gamma_mem, input_mean_mem, input_variance_mem,
          running_mean_mem, running_variance_mem, output_mem, params, backend,
          events);
    } else {
      // Launch forward frozen
      return internal::launch_forward<T, Backend>(
          input_mem, beta_mem, gamma_mem, input_mean_mem, input_variance_mem,
          output_mem, params, backend, events);
    }
  } else {
    auto gradient_mem = backend.get_mem_object(beta_or_gradient, n_items);
    auto beta_grad_mem =
        backend.get_mem_object(running_mean_or_beta_grad, params.channels);
    auto gamma_grad_mem =
        backend.get_mem_object(running_variance_or_gamma_grad, params.channels);
    if (params.is_training) {
      // Launch gradient training
      return internal::launch_gradient<T, Backend>(
          input_mem, gradient_mem, gamma_mem, beta_grad_mem, gamma_grad_mem,
          output_mem, params, backend, events);
    } else {
      auto input_mean_mem = backend.get_mem_object(input_mean, params.channels);
      auto input_variance_mem =
          backend.get_mem_object(input_variance, params.channels);
      // Launch gradient frozen
      return internal::launch_gradient<T, Backend>(
          input_mem, gradient_mem, gamma_mem, input_mean_mem,
          input_variance_mem, beta_grad_mem, gamma_grad_mem, output_mem, params,
          backend, events);
    }
  }
  return StatusCode::InvalidParameter;
}

}  // namespace internal

/**
 * Generic function to launch batchnorm frozen or training, forward or gradient.
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
 * \tparam T The data type of the input tensor.
 * \tparam Backend The type of backend.
 * \tparam Direction The direction of processing, either Forward or Gradient.
 * \param input A pointer to memory representing the input tensor.
 * \param beta_or_gradient A pointer to memory representing the beta tensor for
 *                         forward batchnorm or the gradient tensor for
 *                         gradient batchnorm.
 * \param gamma A pointer to memory representing the gamma tensor.
 * \param input_mean A pointer to memory for input mean tensor. This is ignored
 *                   for gradient training batchnorm.
 * \param input_variance A pointer to memory for input variance tensor. This is
 *                       ignored for gradient training batchnorm.
 * \param running_mean_or_beta_grad A pointer to memory representing the output
 *                                  mean tensor for forward training batchnorm
 *                                  or the beta_grad tensor for gradient
 *                                  batchnorm. This is ignored for forward
 *                                  frozen batchnorm.
 * \param running_variance_or_gamma_grad A pointer to memory representing the
 *                                       output variance tensor for forward
 *                                       training batchnorm or the gamma_grad
 *                                       tensor for gradient batchnorm. This is
 *                                       ignored for forward frozen batchnorm.
 * \param output A pointer to memory representing the output tensor.
 * \param params The batchnorm parameters.
 * \param backend The backend for mapping between pointer representations.
 * \return Returns a SNNStatus containing the SYCL event tied to the kernel
 *         launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, typename Backend, typename Direction,
          typename = typename std::enable_if<
              sycldnn::backend::is_buffer_backend_v<Backend>>::type>
SNNStatus launch(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T const> beta_or_gradient,
    typename Backend::template pointer_type<T const> gamma,
    typename Backend::template pointer_type<T const> input_mean,
    typename Backend::template pointer_type<T const> input_variance,
    typename Backend::template pointer_type<T> running_mean_or_beta_grad,
    typename Backend::template pointer_type<T> running_variance_or_gamma_grad,
    typename Backend::template pointer_type<T> output,
    BatchNormParams const& params, Backend& backend) {
  return internal::sublaunch<T, Backend, Direction>(
      input, beta_or_gradient, gamma, input_mean, input_variance,
      running_mean_or_beta_grad, running_variance_or_gamma_grad, output, params,
      backend, {});
}

/**
 * Generic function to launch batchnorm frozen or training, forward or gradient.
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
 * \tparam T The data type of the input tensor.
 * \tparam Backend The type of backend.
 * \tparam Direction The direction of processing, either Forward or Gradient.
 * \param input A pointer to memory representing the input tensor.
 * \param beta_or_gradient A pointer to memory representing the beta tensor for
 *                         forward batchnorm or the gradient tensor for
 *                         gradient batchnorm.
 * \param gamma A pointer to memory representing the gamma tensor.
 * \param input_mean A pointer to memory for input mean tensor. This is ignored
 *                   for gradient training batchnorm.
 * \param input_variance A pointer to memory for input variance tensor. This is
 *                       ignored for gradient training batchnorm.
 * \param running_mean_or_beta_grad A pointer to memory representing the output
 *                                  mean tensor for forward training batchnorm
 *                                  or the beta_grad tensor for gradient
 *                                  batchnorm. This is ignored for forward
 *                                  frozen batchnorm.
 * \param running_variance_or_gamma_grad A pointer to memory representing the
 *                                       output variance tensor for forward
 *                                       training batchnorm or the gamma_grad
 *                                       tensor for gradient batchnorm. This is
 *                                       ignored for forward frozen batchnorm.
 * \param events     Events which should be completed before the operation
 * \param output A pointer to memory representing the output tensor.
 * \param params The batchnorm parameters.
 * \param backend The backend for mapping between pointer representations.
 * \return Returns a SNNStatus containing the SYCL event tied to the kernel
 *         launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, typename Backend, typename Direction,
          typename = typename std::enable_if<
              sycldnn::backend::is_usm_backend_v<Backend>>::type>
SNNStatus launch(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T const> beta_or_gradient,
    typename Backend::template pointer_type<T const> gamma,
    typename Backend::template pointer_type<T const> input_mean,
    typename Backend::template pointer_type<T const> input_variance,
    typename Backend::template pointer_type<T> running_mean_or_beta_grad,
    typename Backend::template pointer_type<T> running_variance_or_gamma_grad,
    typename Backend::template pointer_type<T> output,
    BatchNormParams const& params, Backend& backend,
    const std::vector<cl::sycl::event>& events = {}) {
  return internal::sublaunch<T, Backend, Direction>(
      input, beta_or_gradient, gamma, input_mean, input_variance,
      running_mean_or_beta_grad, running_variance_or_gamma_grad, output, params,
      backend, events);
}

/**
 * Helper function to launch a forward batchnorm in frozen mode.
 *
 * \tparam T The data type of the input tensor.
 * \tparam Backend The type of backend.
 * \tparam Direction The direction of processing, either Forward or Gradient.
 * \param input A pointer to memory representing the input tensor.
 * \param beta A pointer to memory representing the beta tensor.
 * \param gamma A pointer to memory representing the gamma tensor.
 * \param input_mean A pointer to memory for input mean tensor.
 * \param input_variance A pointer to memory for input variance tensor.
 * \param output A pointer to memory representing the output tensor.
 * \param params The batchnorm parameters.
 * \param backend The backend for mapping between pointer representations.
 * \return Returns a SNNStatus containing the SYCL event tied to the kernel
 *         launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */

template <typename T, typename Backend, typename Direction,
          typename = internal::DisableIfGradient<Direction>,
          typename = typename std::enable_if<
              sycldnn::backend::is_buffer_backend_v<Backend>>::type>
SNNStatus launch(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T const> beta,
    typename Backend::template pointer_type<T const> gamma,
    typename Backend::template pointer_type<T const> input_mean,
    typename Backend::template pointer_type<T const> input_variance,
    typename Backend::template pointer_type<T> output,
    BatchNormParams const& params, Backend& backend) {
  typename Backend::template pointer_type<T> null;
  return internal::sublaunch<T, Backend, Direction>(
      input, beta, gamma, input_mean, input_variance, null, null, output,
      params, backend, {});
}

/**
 * Helper function to launch a forward batchnorm in frozen mode.
 *
 * \tparam T The data type of the input tensor.
 * \tparam Backend The type of backend.
 * \tparam Direction The direction of processing, either Forward or Gradient.
 * \param input A pointer to memory representing the input tensor.
 * \param beta A pointer to memory representing the beta tensor.
 * \param gamma A pointer to memory representing the gamma tensor.
 * \param input_mean A pointer to memory for input mean tensor.
 * \param input_variance A pointer to memory for input variance tensor.
 * \param output A pointer to memory representing the output tensor.
 * \param params The batchnorm parameters.
 * \param backend The backend for mapping between pointer representations.
 * \param events     Events which should be completed before the operation
 * \return Returns a SNNStatus containing the SYCL event tied to the kernel
 *         launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */

template <typename T, typename Backend, typename Direction,
          typename = internal::DisableIfGradient<Direction>,
          typename = typename std::enable_if<
              sycldnn::backend::is_usm_backend_v<Backend>>::type>
SNNStatus launch(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T const> beta,
    typename Backend::template pointer_type<T const> gamma,
    typename Backend::template pointer_type<T const> input_mean,
    typename Backend::template pointer_type<T const> input_variance,
    typename Backend::template pointer_type<T> output,
    BatchNormParams const& params, Backend& backend,
    const std::vector<cl::sycl::event>& events = {}) {
  typename Backend::template pointer_type<T> null;
  return internal::sublaunch<T, Backend, Direction>(
      input, beta, gamma, input_mean, input_variance, null, null, output,
      params, backend, events);
}

/**
 * \cond Doxygen_Suppress
 * Disabling documentation for this function as Doxygen does not differentiate
 * it with the launch above.
 *
 * Helper function to launch a gradient batchnorm in training mode.
 *
 * \tparam T The data type of the input tensor.
 * \tparam Backend The type of backend.
 * \tparam Direction The direction of processing, either Forward or Gradient.
 * \param input A pointer to memory representing the input tensor.
 * \param gradient A pointer to memory representing the gradient tensor.
 * \param gamma A pointer to memory representing the gamma tensor.
 * \param beta_grad A pointer to memory for output beta tensor.
 * \param gamma_grad A pointer to memory for output gamma tensor.
 * \param output A pointer to memory representing the output tensor.
 * \param params The batchnorm parameters.
 * \param backend The backend for mapping between pointer representations.
 * \return Returns a SNNStatus containing the SYCL event tied to the kernel
 *         launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 * \endcond
 */
template <typename T, typename Backend, typename Direction,
          typename = internal::EnableIfGradient<Direction>,
          typename = typename std::enable_if<
              sycldnn::backend::is_buffer_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T const> gradient,
                 typename Backend::template pointer_type<T const> gamma,
                 typename Backend::template pointer_type<T> beta_grad,
                 typename Backend::template pointer_type<T> gamma_grad,
                 typename Backend::template pointer_type<T> output,
                 BatchNormParams const& params, Backend& backend) {
  typename Backend::template pointer_type<T const> null;
  return internal::sublaunch<T, Backend, Direction>(
      input, gradient, gamma, null, null, beta_grad, gamma_grad, output, params,
      backend, {});
}

/**
 * \cond Doxygen_Suppress
 * Disabling documentation for this function as Doxygen does not differentiate
 * it with the launch above.
 *
 * Helper function to launch a gradient batchnorm in training mode.
 *
 * \tparam T The data type of the input tensor.
 * \tparam Backend The type of backend.
 * \tparam Direction The direction of processing, either Forward or Gradient.
 * \param input A pointer to memory representing the input tensor.
 * \param gradient A pointer to memory representing the gradient tensor.
 * \param gamma A pointer to memory representing the gamma tensor.
 * \param beta_grad A pointer to memory for output beta tensor.
 * \param gamma_grad A pointer to memory for output gamma tensor.
 * \param output A pointer to memory representing the output tensor.
 * \param params The batchnorm parameters.
 * \param backend The backend for mapping between pointer representations.
 * \param events     Events which should be completed before the operation
 * \return Returns a SNNStatus containing the SYCL event tied to the kernel
 *         launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 * \endcond
 */
template <typename T, typename Backend, typename Direction,
          typename = internal::EnableIfGradient<Direction>,
          typename = typename std::enable_if<
              sycldnn::backend::is_usm_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T const> gradient,
                 typename Backend::template pointer_type<T const> gamma,
                 typename Backend::template pointer_type<T> beta_grad,
                 typename Backend::template pointer_type<T> gamma_grad,
                 typename Backend::template pointer_type<T> output,
                 BatchNormParams const& params, Backend& backend,
                 const std::vector<cl::sycl::event>& events = {}) {
  typename Backend::template pointer_type<T const> null;
  return internal::sublaunch<T, Backend, Direction>(
      input, gradient, gamma, null, null, beta_grad, gamma_grad, output, params,
      backend, events);
}

}  // namespace batchnorm
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_BATCHNORM_LAUNCH_H_
