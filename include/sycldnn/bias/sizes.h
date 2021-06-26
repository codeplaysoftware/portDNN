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
#ifndef SYCLDNN_INCLUDE_BIAS_SIZES_H_
#define SYCLDNN_INCLUDE_BIAS_SIZES_H_

/**
 * \file
 * Contains functionality for calculating the size of tensors from the
 * bias-add parameters, including the declaration of the
 * \ref sycldnn::bias::BiasSizes structure.
 */
#include "sycldnn/bias/params.h"

#include <cstddef>

namespace sycldnn {
namespace bias {

/** Tensor sizes for a given bias operation. */
struct BiasSizes {
  /** The size of the input tensor in elements. */
  size_t input_size;
  /** The size of the bias tensor in elements. */
  size_t bias_size;
  /** The size of the output tensor in elements. */
  size_t output_size;
};

/**
 * Compute the total sizes of the tensors used in a bias operator for the
 * specified parameters.
 * \param params The bias parameters, containing the tensor sizes and
 *               filter strides.
 * \return Returns a \ref sycldnn::bias::BiasSizes instance, containing
 *         the sizes of the tensors in elements.
 */
inline BiasSizes get_sizes(BiasParams const& params) {
  size_t inp_size =
      params.batch * params.in_rows * params.in_cols * params.channels;

  size_t b_size = params.bias;

  BiasSizes sizes{inp_size, b_size, inp_size};
  return sizes;
}

}  // namespace bias
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_BIAS_SIZES_H_
