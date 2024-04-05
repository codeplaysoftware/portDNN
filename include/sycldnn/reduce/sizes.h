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
#ifndef SYCLDNN_INCLUDE_REDUCE_SIZES_H_
#define SYCLDNN_INCLUDE_REDUCE_SIZES_H_

#include <numeric>

#include "sycldnn/reduce/params.h"

namespace sycldnn {
namespace reduce {

/** Struct that contains the sizes involved in the ND reduction op. */
struct NDReduceSizes {
  /** The type of the sizes is size_t, providing a decent
   * upper bound on the sizes. */
  using Index = size_t;
  /** The total input size of the operation */
  Index input_size;
  /** The total reduction size of the operation */
  Index reduce_size;
  /** The output size at the end of the operation */
  Index output_size;
};

/**
 * \brief Computes the reduction sizes for the specified parameters.
 *
 * \param params The NDReduce parameters, containing the tensor sizes and
 *               axes.
 * \return Returns a \ref sycldnn::reduce::NDReduceSizes instance, containing
 * the sizes of the tensors in elements.
 */
inline NDReduceSizes get_sizes(NDReduceParams const& params) {
  auto const input_size = static_cast<size_t>(std::accumulate(
      params.dims.begin(), params.dims.end(), 1, std::multiplies<int>()));
  size_t reduction_size = 1;
  for (size_t i = 0; i < params.axes.size(); i++) {
    reduction_size *= params.dims[params.axes[i]];
  }
  auto const output_size = input_size / reduction_size;
  return {input_size, reduction_size, output_size};
}
}  // namespace reduce
}  // namespace sycldnn

#endif
