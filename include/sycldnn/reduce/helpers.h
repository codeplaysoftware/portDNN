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
#ifndef SYCLDNN_INCLUDE_REDUCE_HELPERS_H_
#define SYCLDNN_INCLUDE_REDUCE_HELPERS_H_

#include <CL/sycl.hpp>

#include "sycldnn/helpers/macros.h"
#include "sycldnn/mem_object.h"
#include "sycldnn/reduce/operators.h"
#include "sycldnn/reduce/params.h"
#include "sycldnn/reduce/sizes.h"

#include <numeric>

namespace sycldnn {
namespace reduce {

namespace internal {

/** Converts the user-provided *sorted* vector of axes<size_t> to a vector of
 * booleans, which is of the same size as dims, containing true if dimension is
 * to be reduced and false if dimension is kept \param dims Vector of dimensions
 * of input for reduction operation \param axes Vector containing the *sorted*
 * indices of dimensions to be reduced \returns A vector of booleans, which is
 * of the same size as dims, containing true if dimension is to be reduced and
 * false if dimension is kept
 */
inline std::vector<bool> simplify_axes(const std::vector<int>& dims,
                                       const std::vector<int>& axes) {
  std::vector<bool> result;
  int iter = 0;
  for (int i = 0; i < static_cast<int>(dims.size()); i++) {
    if (axes[iter] == i) {
      result.push_back(true);
      iter++;
    } else {
      result.push_back(false);
    }
  }
  return result;
}
/** Squeezes the dimensions passed by the user if two or more consecutive
 * dimensions are being reduced
 * \param dims Vector of dimensions of input for reduction operation
 * \param axes Vector containing the index of which dimension needs to be
 *             reduced
 * \returns A vector of dims where if consecutive dimensions n&m are being
 *          reduced the new value of n is n*m
 */
inline void squeeze_dims_and_axes(const std::vector<int>& dims,
                                  const std::vector<bool> axes_mask,
                                  std::vector<int>& squeezed_dims,
                                  std::vector<int>& squeezed_axis) {
  int new_axis = 0;
  for (size_t i = 0; i < dims.size(); i++) {
    size_t jump = i;
    int new_dim{dims[i]};
    if (dims[i] == 1) {
      continue;
    }
    while (axes_mask[i] && jump + 1 < dims.size() &&
           (axes_mask[jump + 1] || dims[jump + 1] == 1)) {
      new_dim *= dims[jump + 1];
      ++jump;
    }
    if (axes_mask[i]) {
      squeezed_axis.push_back(new_axis);
    }
    i = jump;
    squeezed_dims.push_back(new_dim);
    new_axis++;
  }
  // return result;
}

/** Calculate the output of the first two reductions(required for more than 2
 * non-consecutive reductions)
 * \param dims Vector of dimensions of input for reduction operation
 * \param axes Vector containing the index of which  dimension needs to be
 *             reduced
 * \returns the size of the workspace needed for reduction
 */

inline std::vector<size_t> get_first_two_out_sizes(
    const std::vector<int>& dims, const std::vector<int>& axes) {
  std::vector<size_t> result{};
  auto sorted_axes = axes;
  std::sort(sorted_axes.begin(), sorted_axes.end(), std::greater<size_t>());
  size_t total_size =
      std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());

  auto const size = std::min(sorted_axes.size(), 2UL);
  for (size_t i = 0; i < size; i++) {
    total_size /= dims[sorted_axes[i]];
    result.push_back(total_size);
  }
  return result;
}
}  // namespace internal

/** Calculate the required workspace size for reductions, if the axes are
 * contiguous(get_transformed_axis() returns only one axes of reduction) no
 * workspace is needed returns 0
 * \param dims Vector of dimensions of input for reduction operation
 * \param axes Vector containing the index of which dimension needs to be
 *             reduced
 * \returns the size of the workspace needed for reduction
 */
inline size_t get_workspace_size(NDReduceParams const& params) {
  auto sorted_axes = params.axes;
  std::sort(sorted_axes.begin(), sorted_axes.end());
  auto axes_mask = internal::simplify_axes(params.dims, sorted_axes);

  NDReduceParams squeezed_params;
  internal::squeeze_dims_and_axes(params.dims, axes_mask, squeezed_params.dims,
                                  squeezed_params.axes);

  if (squeezed_params.axes.size() == 1 ||
      sycldnn::reduce::get_sizes(squeezed_params).reduce_size == 1) {
    return 0;
  } else {
    auto out_sizes = internal::get_first_two_out_sizes(squeezed_params.dims,
                                                       squeezed_params.axes);
    return std::accumulate(out_sizes.begin(), out_sizes.end(), 1,
                           std::multiplies<size_t>());
  }
}
}  // namespace reduce
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_REDUCE_HELPERS_H_
