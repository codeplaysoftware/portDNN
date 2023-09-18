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
#ifndef PORTDNN_INCLUDE_POOLING_SIZES_H_
#define PORTDNN_INCLUDE_POOLING_SIZES_H_

/**
 * \file
 * Contains functionality for calculating the size of tensors from the
 * pooling parameters, including the declaration of the
 * \ref sycldnn::pooling::PoolingSizes structure.
 */
#include "portdnn/pooling/operators.h"
#include "portdnn/pooling/params.h"

#include <cstddef>

namespace sycldnn {
namespace pooling {

/** Tensor sizes for a given pooling operation. */
struct PoolingSizes {
  /** The size of the input tensor in elements. */
  size_t input_size;
  /** The size of the output tensor in elements. */
  size_t output_size;
};

/**
 * Compute the total sizes of the tensors used in a pooling operator for the
 * specified parameters.
 * \param params The pooling parameters, containing the tensor sizes and
 *               filter strides.
 * \return Returns a \ref sycldnn::pooling::PoolingSizes instance, containing
 *         the sizes of the tensors in elements.
 */
template <typename PoolingType>
PoolingSizes get_sizes(PoolingParams const& params);

/** \copydoc sycldnn::pooling::get_sizes(PoolingParams const& params) */
template <>
inline PoolingSizes get_sizes<Forward>(PoolingParams const& params) {
  size_t inp_size =
      params.batch * params.in_rows * params.in_cols * params.channels;
  size_t out_size =
      params.batch * params.out_rows * params.out_cols * params.channels;
  PoolingSizes sizes{inp_size, out_size};
  return sizes;
}

/** \copydoc sycldnn::pooling::get_sizes(PoolingParams const& params) */
template <>
inline PoolingSizes get_sizes<Backpropagate>(PoolingParams const& params) {
  size_t inp_size =
      params.batch * params.out_rows * params.out_cols * params.channels;
  size_t out_size =
      params.batch * params.in_rows * params.in_cols * params.channels;
  PoolingSizes sizes{inp_size, out_size};
  return sizes;
}

}  // namespace pooling
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_POOLING_SIZES_H_
