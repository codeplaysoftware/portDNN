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
#ifndef PORTDNN_INCLUDE_SOFTMAX_LAUNCH_H_
#define PORTDNN_INCLUDE_SOFTMAX_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::softmax::launch() function, which
 * asynchronously dispatches a SYCL kernel to compute a softmax operation
 * along a single dimension of a N-dimensional tensor.
 */
#include "portdnn/status.h"

#include "portdnn/softmax/direction.h"
#include "portdnn/softmax/params.h"

#include "portdnn/internal/softmax/launch_internal.h"

#include "portdnn/helpers/macros.h"

namespace sycldnn {
/** Namespace containing the softmax operator. */
namespace softmax {
/** Namespace containing internal implementation details for softmax. */
namespace internal {

/**
 * Validate that the user-provided softmax parameters are consistent with what
 * is expected by portDNN.
 *
 * If compiled with asserts, any invalid parameter will fail with an assert.
 * Otherwise a status code \ref StatusCode::InvalidParameter will be returned.
 *
 * \param params  Softmax parameters to validate.
 * \return        A SNNStatus object containing either \ref StatusCode::OK if
 * all parameters are valid, or \ref StatusCode::InvalidParameter otherwise.
 */
SNNStatus inline validate_params(SoftmaxParams const& params) {
  SNN_VALIDATE_PARAM(params.batch > 0, "The batch size must be positive.");
  SNN_VALIDATE_PARAM(params.channels > 0,
                     "The number of channels/classes must be positive.");
  SNN_VALIDATE_PARAM(params.rows > 0,
                     "The number of input/output rows must be positive.");
  SNN_VALIDATE_PARAM(params.cols > 0,
                     "The number of input/output columns must be positive.");
  return StatusCode::OK;
}

}  // namespace internal

/**
 * Launch the softmax operation kernel in either Forward or Gradient (Backward)
 * direction.
 * Softmax is applied along the channel dimension of a 4D tensor - for 2D
 * matrices with shape (batch x channels), the height and width dimensions can
 * be set to 1.
 *
 * For inputs with height and width > 1, softmax is applied pixel-wise. This
 * is identical to multiplying the batch-size by the total number of pixels for
 * performing softmax on (i.e. batch' = batch x height x width), yielding a 2D
 * matrix as above with dimensions (batch' x channels).
 *
 * \tparam T           The data type of the input tensor.
 * \tparam Direction   The direction of processing, either Forward or Gradient.
 * \tparam Backend     The type of backend.
 * \param input        A pointer to the memory representing the input tensor.
 * \param workspace    A pointer to the memory representing the workspace.
 * \param output       A pointer to the memory representing the output tensor.
 * \param params       The softmax parameters, which describe the tensor shape
 *                     and layout.
 * \param backend      The backend implementation, used to map between pointer
 *                     representations.
 * \return Returns a SNNStatus containing the SYCL event tied to the kernel
 *         launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, typename Direction, typename Backend,
          typename = internal::DisableIfGradient<Direction>,
          typename = typename std::enable_if<
              sycldnn::backend::is_buffer_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T> workspace,
                 typename Backend::template pointer_type<T> output,
                 SoftmaxParams const& params, Backend& backend) {
  auto validation_status = internal::validate_params(params);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }

  return internal::launch<T, Direction>(input, workspace, output, params,
                                        backend, {});
}

/**
 * Launch the softmax operation kernel in either Forward or Gradient (Backward)
 * direction.
 * Softmax is applied along the channel dimension of a 4D tensor - for 2D
 * matrices with shape (batch x channels), the height and width dimensions can
 * be set to 1.
 *
 * For inputs with height and width > 1, softmax is applied pixel-wise. This
 * is identical to multiplying the batch-size by the total number of pixels for
 * performing softmax on (i.e. batch' = batch x height x width), yielding a 2D
 * matrix as above with dimensions (batch' x channels).
 *
 * \tparam T           The data type of the input tensor.
 * \tparam Direction   The direction of processing, either Forward or Gradient.
 * \tparam Backend     The type of backend.
 * \param input        A pointer to the memory representing the input tensor.
 * \param workspace    A pointer to the memory representing the workspace.
 * \param output       A pointer to the memory representing the output tensor.
 * \param params       The softmax parameters, which describe the tensor shape
 *                     and layout.
 * \param backend      The backend implementation, used to map between pointer
 *                     representations.
 * \param events       Events which should be completed before the operation.
 * \return Returns a SNNStatus containing the SYCL event tied to the kernel
 *         launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, typename Direction, typename Backend,
          typename = internal::DisableIfGradient<Direction>,
          typename = typename std::enable_if<
              sycldnn::backend::is_usm_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T> workspace,
                 typename Backend::template pointer_type<T> output,
                 SoftmaxParams const& params, Backend& backend,
                 const std::vector<cl::sycl::event>& events = {}) {
  auto validation_status = internal::validate_params(params);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }

  return internal::launch<T, Direction>(input, workspace, output, params,
                                        backend, events);
}

/**
 * Launch the softmax operation kernel in either Gradient (Backward)
 * direction.
 * \tparam T           The data type of the input tensor.
 * \tparam Direction   The direction of processing, either Forward or Gradient.
 * \tparam Backend     The type of backend.
 * \param input        A pointer to the memory representing the input tensor.
 * \param gradient     A pointer to the memory representing the gradient tensor.
 * \param workspace    A pointer to the memory representing the workspace.
 * \param output       A pointer to the memory representing the output tensor.
 * \param params       The softmax parameters, which describe the tensor shape
 *                     and layout.
 * \param backend      The backend implementation, used to map between pointer
 *                     representations.
 * \return Returns a SNNStatus containing the SYCL event tied to the kernel
 *         launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */

template <typename T, typename Direction, typename Backend,
          typename = internal::EnableIfGradient<Direction>,
          typename = typename std::enable_if<
              sycldnn::backend::is_buffer_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T const> gradient,
                 typename Backend::template pointer_type<T> workspace,
                 typename Backend::template pointer_type<T> output,
                 SoftmaxParams const& params, Backend& backend) {
  auto validation_status = internal::validate_params(params);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }

  return internal::launch<T, Direction>(input, gradient, workspace, output,
                                        params, backend, {});
}

/**
 * Launch the softmax operation kernel in either Gradient (Backward)
 * direction.
 * \tparam T           The data type of the input tensor.
 * \tparam Direction   The direction of processing, either Forward or Gradient.
 * \tparam Backend     The type of backend.
 * \param input        A pointer to the memory representing the input tensor.
 * \param gradient     A pointer to the memory representing the gradient tensor.
 * \param workspace    A pointer to the memory representing the workspace.
 * \param output       A pointer to the memory representing the output tensor.
 * \param params       The softmax parameters, which describe the tensor shape
 *                     and layout.
 * \param backend      The backend implementation, used to map between pointer
 *                     representations.
 * \param events       Events which should be completed before the operation.
 * \return Returns a SNNStatus containing the SYCL event tied to the kernel
 *         launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */

template <typename T, typename Direction, typename Backend,
          typename = internal::EnableIfGradient<Direction>,
          typename = typename std::enable_if<
              sycldnn::backend::is_usm_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T const> gradient,
                 typename Backend::template pointer_type<T> workspace,
                 typename Backend::template pointer_type<T> output,
                 SoftmaxParams const& params, Backend& backend,
                 const std::vector<cl::sycl::event>& events = {}) {
  auto validation_status = internal::validate_params(params);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }

  return internal::launch<T, Direction>(input, gradient, workspace, output,
                                        params, backend, events);
}

}  // namespace softmax
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_SOFTMAX_LAUNCH_H_
