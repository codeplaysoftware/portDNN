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

#include "sycldnn/batchnorm/operation.h"
#include "sycldnn/batchnorm/params.h"

#include "sycldnn/internal/batchnorm/launch_internal.h"

#include "sycldnn/helpers/macros.h"

namespace sycldnn {
/** Namespace containing the batchnorm operator. */
namespace batchnorm {
/** Namespace containing internal implementation details for batchnorm. */
namespace internal {

/**
 * Validate that the user-provided batchnorm parameters are consistent with what
 * is expected by SYCL-DNN.
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
  SNN_VALIDATE_PARAM(params.input_format == sycldnn::DataFormat::NHWC,
                     "Currently SYCL-DNN only supports the NHWC data format.");
  SNN_VALIDATE_PARAM(params.epsilon > 0.f,
                     "The epsilon parameter must be greater than 0.");
  SNN_VALIDATE_PARAM(
      params.momentum >= 0.f,
      "The momentum parameter must be greater than or equal to 0.");
  return SNNStatus{{}, StatusCode::OK};
}

}  // namespace internal

/**
 * Launch the batchnorm operation kernel in the forward direction when computing
 * the Mean and Variance for Batchnorm Computation.
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
 * \tparam T           The data type of the input tensor.
 * \tparam Backend     The type of backend.
 * \tparam Direction   Either Forward or Gradient.
 * \tparam Operation   Either Training or Frozen.
 * \param input        A pointer to memory representing the input tensor.
 * \param beta         A pointer to memory representing the beta tensor.
 * \param gamma        A pointer to memory representing the gamma tensor.
 * \param input_mean   A pointer to memory for input mean tensor.
 * \param input_variance A pointer to memory for input variance tensor.
 * \param running_mean  A pointer to memory for output mean tensor.
 * \param running_variance A pointer to memory for output variance tensor.
 * \param output       A pointer to memory representing the output tensor.
 * \param params       The batchnorm parameters.
 * \param backend      The backend for mapping between pointer representations.
 * \return             Returns a SNNStatus containing the SYCL event tied to
 *                     the kernel launches and a StatusCode enum showing if the
 *                     launch was OK or whether it encountered some problem.
 */
template <typename T, typename Backend, typename Direction, typename Operation,
          typename = internal::EnableIf_Forward_Training<Direction, Operation>>
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
  auto validation_status = internal::validate_params(params);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }

  return internal::launch_forward<T, Backend, Direction, Operation>(
      input, beta, gamma, input_mean, input_variance, running_mean,
      running_variance, output, params, backend);
}

/**
 * Launch the batchnorm operation kernel in the forward direction when using the
 * existing Mean and Variance for Batchnorm Computation.
 *
 * \tparam T           The data type of the input tensor.
 * \tparam Backend     The type of backend.
 * \tparam Direction   Either Forward or Gradient
 * \tparam Operation   Either Training or Frozen.
 * \param input        A pointer to memory representing the input tensor.
 * \param beta         A pointer to memory representing the beta tensor.
 * \param gamma        A pointer to memory representing the gamma tensor.
 * \param input_mean  A pointer to memory for input mean tensor.
 * \param input_variance A pointer to memory for input variance tensor.
 * \param output       A pointer to memory representing the output tensor.
 * \param params       The batchnorm parameters.
 * \param backend      The backend for mapping between pointer representations.
 * \return             Returns a SNNStatus containing the SYCL event tied to
 *                     the kernel launches and a StatusCode enum showing if the
 *                     launch was OK or whether it encountered some problem.
 */

template <typename T, typename Backend, typename Direction, typename Operation,
          typename = internal::DisableIf_Forward_Training<Direction, Operation>>
SNNStatus launch_forward(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T const> beta,
    typename Backend::template pointer_type<T const> gamma,
    typename Backend::template pointer_type<T const> input_mean,
    typename Backend::template pointer_type<T const> input_variance,
    typename Backend::template pointer_type<T> output,
    BatchNormParams const& params, Backend& backend) {
  auto validation_status = internal::validate_params(params);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }

  auto n_items = params.batch * params.channels * params.rows * params.cols;

  auto queue = backend.get_queue();

  auto in_mem = backend.get_mem_object(input, n_items);
  auto out_mem = backend.get_mem_object(output, n_items);

  auto beta_mem = backend.get_mem_object(beta, params.channels);
  auto gamma_mem = backend.get_mem_object(gamma, params.channels);
  auto mean_mem = backend.get_mem_object(input_mean, params.channels);
  auto variance_mem = backend.get_mem_object(input_variance, params.channels);

  return internal::launch_forward<T, Direction, Operation>(
      in_mem, beta_mem, gamma_mem, mean_mem, variance_mem, out_mem, params,
      queue);
}

/**
 * Launch the batchnorm operation kernel in the Gradient direction when
 * computing the Mean and Variance for Batchnorm Computation.
 *
 * \tparam T           The data type of the input tensor.
 * \tparam Backend     The type of backend.
 * \tparam Direction   Either Forward or Gradient.
 * \tparam Operation   Either Training or Frozen.
 * \param input        A pointer to memory representing the input tensor.
 * \param gradient     A pointer to memory representing the gradient tensor.
 * \param gamma        A pointer to memory representing the gamma tensor.
 * \param workspace    A pointer to memory representing the workspace.
 * \param beta_grad    A pointer to memory for output beta tensor.
 * \param gamma_grad   A pointer to memory for output gamma tensor.
 * \param output       A pointer to memory representing the output tensor.
 * \param params       The batchnorm parameters.
 * \param backend      The backend for mapping between pointer representations.
 * \return             Returns a SNNStatus containing the SYCL event tied to
 *                     the kernel launches and a StatusCode enum showing if the
 *                     launch was OK or whether it encountered some problem.
 */
template <typename T, typename Backend, typename Direction, typename Operation,
          typename = internal::EnableIf_Gradient_Training<Direction, Operation>>
SNNStatus launch_grad(typename Backend::template pointer_type<T const> input,
                      typename Backend::template pointer_type<T const> gradient,
                      typename Backend::template pointer_type<T const> gamma,
                      typename Backend::template pointer_type<T> workspace,
                      typename Backend::template pointer_type<T> beta_grad,
                      typename Backend::template pointer_type<T> gamma_grad,
                      typename Backend::template pointer_type<T> output,
                      BatchNormParams const& params, Backend& backend) {
  auto validation_status = internal::validate_params(params);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }

  return internal::launch_grad<T, Backend, Direction, Operation>(
      input, gradient, gamma, workspace, beta_grad, gamma_grad, output, params,
      backend);
}

/**
 * Launch the batchnorm operation kernel in the Gradient direction when using
 * the existing Mean and Variance for Batchnorm Computation.
 *
 * \tparam T           The data type of the input tensor.
 * \tparam Backend     The type of backend.
 * \tparam Direction   Either Forward or Gradient
 * \tparam Operation   Either Training or Frozen.
 * \param input        A pointer to memory representing the input tensor.
 * \param gradient     A pointer to memory representing the gradient tensor.
 * \param gamma        A pointer to memory representing the gamma tensor.
 * \param pop_mean     A pointer to memory for input mean tensor.
 * \param pop_variance A pointer to memory for input variance tensor.
 * \param beta_grad    A pointer to memory for output beta tensor.
 * \param gamma_grad   A pointer to memory for output gamma tensor.
 * \param output       A pointer to memory representing the output tensor.
 * \param params       The batchnorm parameters.
 * \param backend      The backend for mapping between pointer representations.
 * \return             Returns a SNNStatus containing the SYCL event tied to
 *                     the kernel launches and a StatusCode enum showing if the
 *                     launch was OK or whether it encountered some problem.
 */
template <
    typename T, typename Backend, typename Direction, typename Operation,
    typename = internal::DisableIf_Gradient_Training<Direction, Operation>>
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
  auto validation_status = internal::validate_params(params);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }

  return internal::launch_grad<T, Backend, Direction, Operation>(
      input, gradient, gamma, pop_mean, pop_variance, beta_grad, gamma_grad,
      output, params, backend);
}

}  // namespace batchnorm
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_BATCHNORM_LAUNCH_H_
