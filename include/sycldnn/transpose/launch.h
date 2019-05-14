/*
 * Copyright 2019 Codeplay Software Ltd
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
#ifndef SYCLDNN_INCLUDE_TRANSPOSE_LAUNCH_H_
#define SYCLDNN_INCLUDE_TRANSPOSE_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::transpose::launch() function, which
 * asynchronously dispatches a SYCL kernel to transpose an N-Dimensional
 * tensor.
 */
#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/helpers/macros.h"

#include "sycldnn/internal/transpose/launch.h"

#include <vector>

namespace sycldnn {
/** Namespace containing tensor transpose operations. */
namespace transpose {
/**
 * Transpose an ND tensor using any permutation of the dimensions.
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
template <typename T, typename Backend>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T> output,
                 std::vector<int> const& dimensions,
                 std::vector<int> const& permutation, Backend& backend) {
  auto n_dimensions = dimensions.size();
  SNN_VALIDATE_PARAM(n_dimensions > 0u,
                     "The number of dimensions must be positive.");
  SNN_VALIDATE_PARAM(n_dimensions < 7u,
                     "Only dimensions 6 and fewer are supported.");
  SNN_VALIDATE_PARAM(permutation.size() == n_dimensions,
                     "The number of permutations entries must match the number "
                     "of dimensions.");

  std::vector<bool> not_seen(n_dimensions, true);
  for (int value : permutation) {
    SNN_VALIDATE_PARAM(value >= 0 && static_cast<size_t>(value) < n_dimensions,
                       "Each permutation value must index a dimension.");
    SNN_VALIDATE_PARAM(not_seen[value],
                       "Each permutation value must be distinct.");
    not_seen[value] = false;
  }

  size_t tensor_size = std::accumulate(begin(dimensions), end(dimensions),
                                       static_cast<size_t>(1),
                                       [](size_t a, int b) { return a * b; });
  SNN_VALIDATE_PARAM(tensor_size > 0, "Tensor size must be positive.");

  auto in_buff = backend.get_buffer(input, tensor_size);
  auto out_buff = backend.get_buffer(output, tensor_size);

  size_t in_offset = backend.get_offset(input);
  size_t out_offset = backend.get_offset(output);

  auto in_acc = make_mem_object(in_buff, tensor_size, in_offset);
  auto out_acc = make_mem_object(out_buff, tensor_size, out_offset);

  auto sycl_queue = backend.get_queue();

  return internal::launch<T>(in_acc, out_acc, dimensions, permutation,
                             sycl_queue);
}

/**
 * Convert an NHWC tensor to an NCHW tensor.
 *
 * \param input       A pointer to the memory representing the input tensor.
 * \param output      A pointer to the memory representing the output tensor.
 * \param dimensions  Number of elements in each dimension of the input. The
 *                    size of the vector should match the number of dimensions
 *                    in the input tensor.
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
template <typename T, typename Backend>
SNNStatus convert_nhwc_to_nchw(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T> output,
    std::vector<int> const& dimensions, Backend& backend) {
  SNN_VALIDATE_PARAM(
      dimensions.size() == 4,
      "Conversion from NHWC to NCHW is only valid on 4D tensors.");
  return launch<T>(input, output, dimensions, {0, 3, 1, 2}, backend);
}

/**
 * Convert an NCHW tensor to an NHWC tensor.
 *
 * \param input       A pointer to the memory representing the input tensor.
 * \param output      A pointer to the memory representing the output tensor.
 * \param dimensions  Number of elements in each dimension of the input. The
 *                    size of the vector should match the number of dimensions
 *                    in the input tensor.
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
template <typename T, typename Backend>
SNNStatus convert_nchw_to_nhwc(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T> output,
    std::vector<int> const& dimensions, Backend& backend) {
  SNN_VALIDATE_PARAM(
      dimensions.size() == 4,
      "Conversion from NCHW to NHWC is only valid on 4D tensors.");
  return launch<T>(input, output, dimensions, {0, 2, 3, 1}, backend);
}

}  // namespace transpose
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_TRANSPOSE_LAUNCH_H_
