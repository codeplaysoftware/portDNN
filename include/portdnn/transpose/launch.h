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
#ifndef PORTDNN_INCLUDE_TRANSPOSE_LAUNCH_H_
#define PORTDNN_INCLUDE_TRANSPOSE_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::transpose::launch() function, which
 * asynchronously dispatches a SYCL kernel to transpose an N-Dimensional
 * tensor.
 */
#include "portdnn/backend/backend_helpers.h"
#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/helpers/macros.h"

#include "portdnn/internal/transpose/launch.h"

#include <vector>

namespace sycldnn {
/** Namespace containing tensor transpose operations. */
namespace transpose {
/**
 * Transpose an ND tensor using any permutation of the input dimensions.
 *
 * \param input       A pointer to the memory representing the input tensor.
 * \param output      A pointer to the memory representing the output tensor.
 * \param dimensions  Number of elements in each dimension. The size of the
 *                    vector should match the number of dimensions in the input
 *                    tensor.
 * \param permutation A vector of zero indexed integers representing the
 *                    permutation to use in the transpose. The i-th dimension
 *                    of the output will be mapped to the permutation[i]-th
 *                    dimension in the input.
 * \param backend     The backend implementation, used to map between pointer
 *                    representations.
 * \return Returns an SNNStatus containing the SYCL event tied to the kernel
 *         launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 * \retval StatusCode::InvalidParameter: An invalid parameter was passed in to
 *         the launch function:
 *         * The size of dimension was zero or over 6.
 *         * The size of dimension doesn't match the size of permutation.
 *         * A value in permutation doesn't map to a dimension.
 *         * The tensor size was zero.
 * \retval StatusCode::OK: The kernel was launched successfully.
 */
template <typename T, typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_buffer_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T> output,
                 std::vector<int> const& dimensions,
                 std::vector<int> const& permutation, Backend& backend) {
  return internal::sublaunch<T>(input, output, dimensions, permutation, backend,
                                {});
}

/**
 * Transpose an ND tensor using any permutation of the input dimensions.
 *
 * \param input       A pointer to the memory representing the input tensor.
 * \param output      A pointer to the memory representing the output tensor.
 * \param dimensions  Number of elements in each dimension. The size of the
 *                    vector should match the number of dimensions in the input
 *                    tensor.
 * \param permutation A vector of zero indexed integers representing the
 *                    permutation to use in the transpose. The i-th dimension
 *                    of the output will be mapped to the permutation[i]-th
 *                    dimension in the input.
 * \param backend     The backend implementation, used to map between pointer
 *                    representations.
 * \param events     Events which should be completed before the operation
 * executes. \return Returns an SNNStatus containing the SYCL event tied to the
 * kernel launches and a StatusCode enum showing if the launch was OK or whether
 * it encountered some problem. \retval StatusCode::InvalidParameter: An invalid
 * parameter was passed in to the launch function:
 *         * The size of dimension was zero or over 6.
 *         * The size of dimension doesn't match the size of permutation.
 *         * A value in permutation doesn't map to a dimension.
 *         * The tensor size was zero.
 * \retval StatusCode::OK: The kernel was launched successfully.
 */
template <typename T, typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_usm_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T> output,
                 std::vector<int> const& dimensions,
                 std::vector<int> const& permutation, Backend& backend,
                 const std::vector<cl::sycl::event>& events = {}) {
  return internal::sublaunch<T>(input, output, dimensions, permutation, backend,
                                events);
}

/**
 * Convert an NHWC tensor to an NCHW tensor.
 *
 * \param input       A pointer to the memory representing the input tensor.
 * \param output      A pointer to the memory representing the output tensor.
 * \param dimensions  Number of elements in each dimension. The size of the
 *                    vector should match the number of dimensions in the input
 *                    tensor.
 * \param backend     The backend implementation, used to map between pointer
 *                    representations.
 * \return Returns an SNNStatus containing the SYCL event tied to the kernel
 *         launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 * \retval StatusCode::InvalidParameter: An invalid parameter was passed in to
 *         the launch function:
 *         * The number of dimensions was not 4.
 *         * The tensor size was zero.
 * \retval StatusCode::OK: The kernel was launched successfully.
 */
template <typename T, typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_buffer_backend_v<Backend>>::type>
SNNStatus convert_nhwc_to_nchw(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T> output,
    std::vector<int> const& dimensions, Backend& backend) {
  SNN_VALIDATE_PARAM(
      dimensions.size() == 4,
      "Conversion from NHWC to NCHW is only valid on 4D tensors.");
  return internal::sublaunch<T>(input, output, dimensions, NHWC_TO_NCHW,
                                backend, {});
}

/**
 * Convert an NHWC tensor to an NCHW tensor.
 *
 * \param input       A pointer to the memory representing the input tensor.
 * \param output      A pointer to the memory representing the output tensor.
 * \param dimensions  Number of elements in each dimension. The size of the
 *                    vector should match the number of dimensions in the input
 *                    tensor.
 * \param backend     The backend implementation, used to map between pointer
 *                    representations.
 * \param events     Events which should be completed before the operation
 * executes. \return Returns an SNNStatus containing the SYCL event tied to the
 * kernel launches and a StatusCode enum showing if the launch was OK or whether
 * it encountered some problem. \retval StatusCode::InvalidParameter: An invalid
 * parameter was passed in to the launch function:
 *         * The number of dimensions was not 4.
 *         * The tensor size was zero.
 * \retval StatusCode::OK: The kernel was launched successfully.
 */
template <typename T, typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_usm_backend_v<Backend>>::type>
SNNStatus convert_nhwc_to_nchw(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T> output,
    std::vector<int> const& dimensions, Backend& backend,
    const std::vector<cl::sycl::event>& events = {}) {
  SNN_VALIDATE_PARAM(
      dimensions.size() == 4,
      "Conversion from NHWC to NCHW is only valid on 4D tensors.");
  return internal::sublaunch<T>(input, output, dimensions, NHWC_TO_NCHW,
                                backend, events);
}

/**
 * Convert an NCHW tensor to an NHWC tensor.
 *
 * \param input       A pointer to the memory representing the input tensor.
 * \param output      A pointer to the memory representing the output tensor.
 * \param dimensions  Number of elements in each dimension. The size of the
 *                    vector should match the number of dimensions in the input
 *                    tensor.
 * \param backend     The backend implementation, used to map between pointer
 *                    representations.
 * \return Returns an SNNStatus containing the SYCL event tied to the kernel
 *         launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 * \retval StatusCode::InvalidParameter: An invalid parameter was passed in to
 *         the launch function:
 *         * The number of dimensions was not 4.
 *         * The tensor size was zero.
 * \retval StatusCode::OK: The kernel was launched successfully.
 */
template <typename T, typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_buffer_backend_v<Backend>>::type>
SNNStatus convert_nchw_to_nhwc(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T> output,
    std::vector<int> const& dimensions, Backend& backend) {
  SNN_VALIDATE_PARAM(
      dimensions.size() == 4,
      "Conversion from NCHW to NHWC is only valid on 4D tensors.");
  return internal::sublaunch<T>(input, output, dimensions, NCHW_TO_NHWC,
                                backend, {});
}

/**
 * Convert an NCHW tensor to an NHWC tensor.
 *
 * \param input       A pointer to the memory representing the input tensor.
 * \param output      A pointer to the memory representing the output tensor.
 * \param dimensions  Number of elements in each dimension. The size of the
 *                    vector should match the number of dimensions in the input
 *                    tensor.
 * \param backend     The backend implementation, used to map between pointer
 *                    representations.
 * \param events     Events which should be completed before the operation
 * executes. \return Returns an SNNStatus containing the SYCL event tied to the
 * kernel launches and a StatusCode enum showing if the launch was OK or whether
 * it encountered some problem. \retval StatusCode::InvalidParameter: An invalid
 * parameter was passed in to the launch function:
 *         * The number of dimensions was not 4.
 *         * The tensor size was zero.
 * \retval StatusCode::OK: The kernel was launched successfully.
 */
template <typename T, typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_usm_backend_v<Backend>>::type>
SNNStatus convert_nchw_to_nhwc(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T> output,
    std::vector<int> const& dimensions, Backend& backend,
    const std::vector<cl::sycl::event>& events = {}) {
  SNN_VALIDATE_PARAM(
      dimensions.size() == 4,
      "Conversion from NCHW to NHWC is only valid on 4D tensors.");
  return internal::sublaunch<T>(input, output, dimensions, NCHW_TO_NHWC,
                                backend, events);
}

}  // namespace transpose
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_TRANSPOSE_LAUNCH_H_
