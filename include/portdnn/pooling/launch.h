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

#ifndef PORTDNN_INCLUDE_POOLING_LAUNCH_H_
#define PORTDNN_INCLUDE_POOLING_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::pooling::launch() function, which asynchronously
 * dispatches the SYCL kernels to compute a 2D pooling operation.
 */

#include "portdnn/backend/backend_helpers.h"
#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/helpers/macros.h"

#include "portdnn/pooling/operators.h"
#include "portdnn/pooling/params.h"
#include "portdnn/pooling/sizes.h"

#include "portdnn/internal/pooling/launch_internal.h"

namespace sycldnn {
/** Namespace containing all pooling operations. */
namespace pooling {
/** Namespace containing internal implementation details for pooling. */
namespace internal {

/**
 * Validate that the user provided pooling parameters are consistent with what
 * is expected by portDNN.
 *
 * If compiled with asserts, any invalid parameter will fail an assert.
 * Otherwise a status code \ref StatusCode::InvalidParameter will be returned.
 *
 * \tparam Direction  Forward or Backprop
 * \param [in] params User provided parameters to validate
 * \return An SNNStatus object containing either \ref StatusCode::OK if all
 *         parameters are valid, or \ref StatusCode::InvalidParameter otherwise.
 */
template <typename Direction>
SNNStatus inline validate_params(PoolingParams const& params) {
  SNN_VALIDATE_PARAM(params.batch > 0, "The batch size must be positive.");
  SNN_VALIDATE_PARAM(params.channels > 0,
                     "The number of channels must be positive.");
  SNN_VALIDATE_PARAM(params.in_rows > 0,
                     "The number of input rows must be positive.");
  SNN_VALIDATE_PARAM(params.in_cols > 0,
                     "The number of input columns must be positive.");
  SNN_VALIDATE_PARAM(params.out_rows > 0,
                     "The number of output rows must be positive.");
  SNN_VALIDATE_PARAM(params.out_cols > 0,
                     "The number of output columns must be positive.");
  SNN_VALIDATE_PARAM(params.window_rows > 0,
                     "The number of window rows must be positive.");
  SNN_VALIDATE_PARAM(params.window_cols > 0,
                     "The number of window columns must be positive.");
  SNN_VALIDATE_PARAM(params.stride_rows > 0,
                     "The stride in the row direction must be positive.");
  SNN_VALIDATE_PARAM(params.stride_cols > 0,
                     "The stride in the column direction must be positive.");
  SNN_VALIDATE_PARAM(params.pad_rows >= 0,
                     "The padding in the row direction must be non-negative.");
  SNN_VALIDATE_PARAM(
      params.pad_cols >= 0,
      "The padding in the column direction must be non-negative.");
  SNN_VALIDATE_PARAM(
      params.input_format == sycldnn::DataFormat::NHWC ||
          (params.input_format == sycldnn::DataFormat::NCHW &&
           std::is_same<Direction, Forward>::value),
      "Currently portDNN pooling supports the NHWC and NCHW data formats.");
  return StatusCode::OK;
}

}  // namespace internal

/**
 * Launch the pooling operation kernel.
 *
 * \tparam T         The data type of the input tensor.
 * \tparam PoolType  The type of pooling used depends on the PoolType template
 *                   parameter, which can be used to specify either Max or
 *                   Average pooling.
 * \tparam Direction Whether the pooling operation computed should be the
 *                   Forward or Backpropagate pass.
 * \tparam Backend   The type of the Backend.
 *
 * \param [in]  input    A pointer to the input tensor.
 * \param [out] output   A pointer to the output tensor.
 * \param [in]  pp       The parameters of the pooling operation.
 * \param [in]  backend  The backend that provides access to the SYCL buffers
 *                       corresponding to the input and output pointers.
 * \return An \ref SNNStatus containing the SYCL event tied to the kernel
 *         launches and a \ref StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, template <typename> class PoolType, typename Direction,
          typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_buffer_backend_v<Backend> &&
                  !internal::IsMaxGradient<T, PoolType, Direction>::value,
              int>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T> output,
                 const PoolingParams& pp, Backend& backend) {
  auto validation_status = internal::validate_params<Direction>(pp);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }

  return internal::sublaunch<T, PoolType, Direction, Backend>(input, output, pp,
                                                              backend, {});
}

/**
 * Launch the max pooling gradient kernel.
 *
 * \tparam T         The data type of the input tensor.
 * \tparam PoolType  The type of pooling used depends on the PoolType template
 *                   parameter, which can be used to specify either Max or
 *                   Average pooling.
 * \tparam Direction Whether the pooling operation computed should be the
 *                   Forward or Backpropagate pass.
 * \tparam Backend   The type of the Backend.
 *
 * \param [in]  input_data     A pointer to the original input tensor.
 * \param [in]  output_data    A pointer to the original output tensor.
 * \param [in]  input_backprop A pointer to the backprop error tensor.
 * \param [out] output         A pointer to the output tensor.
 * \param [in]  pp             The parameters of the pooling operation.
 * \param [in]  backend        The backend that provides access to the SYCL
 *                             buffers corresponding to the input and output
 *                             pointers.
 * \return An \ref SNNStatus containing the SYCL event tied to the kernel
 *         launches and a \ref StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, template <typename> class PoolType, typename Direction,
          typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_buffer_backend_v<Backend> &&
                  internal::IsMaxGradient<T, PoolType, Direction>::value,
              int>::type>
SNNStatus launch(
    typename Backend::template pointer_type<T const> input_data,
    typename Backend::template pointer_type<T const> output_data,
    typename Backend::template pointer_type<T const> input_backprop,
    typename Backend::template pointer_type<T> output, const PoolingParams& pp,
    Backend& backend) {
  auto validation_status = internal::validate_params<Direction>(pp);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }

  return internal::sublaunch<T, PoolType, Direction, Backend>(
      input_data, output_data, input_backprop, output, pp, backend, {});
}

/**
 * Launch the pooling operation kernel.
 *
 * \tparam T         The data type of the input tensor.
 * \tparam PoolType  The type of pooling used depends on the PoolType template
 *                   parameter, which can be used to specify either Max or
 *                   Average pooling.
 * \tparam Direction Whether the pooling operation computed should be the
 *                   Forward or Backpropagate pass.
 * \tparam Backend   The type of the Backend.
 *
 * \param [in]  input    A pointer to the input tensor.
 * \param [out] output   A pointer to the output tensor.
 * \param [in]  pp       The parameters of the pooling operation.
 * \param [in]  backend  The backend that provides access to the SYCL buffers
 *                       corresponding to the input and output pointers.
 * \param [in] events    USM dependency events.
 * \return An \ref SNNStatus containing the SYCL event tied to the kernel
 *         launches and a \ref StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, template <typename> class PoolType, typename Direction,
          typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_usm_backend_v<Backend> &&
                  !internal::IsMaxGradient<T, PoolType, Direction>::value,
              int>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T> output,
                 const PoolingParams& pp, Backend& backend,
                 const std::vector<cl::sycl::event>& events = {}) {
  auto validation_status = internal::validate_params<Direction>(pp);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }

  return internal::sublaunch<T, PoolType, Direction, Backend>(input, output, pp,
                                                              backend, events);
}

/**
 * Launch the max pooling gradient kernel.
 *
 * \tparam T         The data type of the input tensor.
 * \tparam PoolType  The type of pooling used depends on the PoolType template
 *                   parameter, which can be used to specify either Max or
 *                   Average pooling.
 * \tparam Direction Whether the pooling operation computed should be the
 *                   Forward or Backpropagate pass.
 * \tparam Backend   The type of the Backend.
 *
 * \param [in]  input_data     A pointer to the original input tensor.
 * \param [in]  output_data    A pointer to the original output tensor.
 * \param [in]  input_backprop A pointer to the backprop error tensor.
 * \param [out] output         A pointer to the output tensor.
 * \param [in]  pp             The parameters of the pooling operation.
 * \param [in]  backend        The backend that provides access to the SYCL
 *                             buffers corresponding to the input and output
 *                             pointers.
 * \param [in] events          USM dependency events.
 * \return An \ref SNNStatus containing the SYCL event tied to the kernel
 *         launches and a \ref StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, template <typename> class PoolType, typename Direction,
          typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_usm_backend_v<Backend> &&
                  internal::IsMaxGradient<T, PoolType, Direction>::value,
              int>::type>
SNNStatus launch(
    typename Backend::template pointer_type<T const> input_data,
    typename Backend::template pointer_type<T const> output_data,
    typename Backend::template pointer_type<T const> input_backprop,
    typename Backend::template pointer_type<T> output, const PoolingParams& pp,
    Backend& backend, const std::vector<cl::sycl::event>& events = {}) {
  auto validation_status = internal::validate_params<Direction>(pp);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }

  return internal::sublaunch<T, PoolType, Direction, Backend>(
      input_data, output_data, input_backprop, output, pp, backend, events);
}

}  // namespace pooling
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_POOLING_LAUNCH_H_
