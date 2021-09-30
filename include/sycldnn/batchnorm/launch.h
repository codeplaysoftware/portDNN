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
  SNN_VALIDATE_PARAM(params.epsilon > float(0),
                     "The epsilon parameter must be greater than 0.");
  return SNNStatus{{}, StatusCode::OK};
}

}  // namespace internal

/**
 * Launch the batchnorm operation kernel in the forward direction.
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
 * \tparam Operation   Either Training or Inference.
 * \param input        A pointer to memory representing the input tensor.
 * \param beta         A pointer to memory representing the beta tensor.
 * \param gamma        A pointer to memory representing the gamma tensor.
 * \param moving_mean  A pointer to memory for moving_mean tensor.
 * \param moving_variance A pointer to memory for moving_variance tensor.
 * \param output       A pointer to memory representing the output tensor.
 * \param params       The batchnorm parameters.
 * \param backend      The backend for mapping between pointer representations.
 * \return             Returns a SNNStatus containing the SYCL event tied to
 *                     the kernel launches and a StatusCode enum showing if the
 *                     launch was OK or whether it encountered some problem.
 */
template <typename T, typename Backend, typename Operation,
          typename = internal::EnableIfTraining<Operation>>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T const> beta,
                 typename Backend::template pointer_type<T const> gamma,
                 typename Backend::template pointer_type<T> moving_mean,
                 typename Backend::template pointer_type<T> moving_variance,
                 typename Backend::template pointer_type<T> output,
                 BatchNormParams const& params, Backend& backend) {
  auto validation_status = internal::validate_params(params);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }

  return internal::launch_batchnorm_training<T, Backend, Operation>(
      input, beta, gamma, moving_mean, moving_variance, output, params,
      backend);
}

template <typename T, typename Backend, typename Operation,
          typename = internal::DisableIfTraining<Operation>>
SNNStatus launch(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T const> beta,
    typename Backend::template pointer_type<T const> gamma,
    typename Backend::template pointer_type<T const> moving_mean,
    typename Backend::template pointer_type<T const> moving_variance,
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
  auto mean_mem = backend.get_mem_object(moving_mean, params.channels);
  auto variance_mem = backend.get_mem_object(moving_variance, params.channels);

  return internal::launch_batchnorm_inference<T, Operation>(
      in_mem, beta_mem, gamma_mem, mean_mem, variance_mem, out_mem, params,
      queue);
}

}  // namespace batchnorm
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_BATCHNORM_LAUNCH_H_
