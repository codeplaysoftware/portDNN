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
#ifndef PORTDNN_INCLUDE_GATHER_SIZES_H_
#define PORTDNN_INCLUDE_GATHER_SIZES_H_

/**
 * \file
 * Contains functionality for calculating the size of tensors and blocks
 * from the Gather parameters, including the declaration of the
 * \ref sycldnn::gather::GatherSizes structure.
 */
#include <functional>
#include <numeric>

#include "portdnn/gather/params.h"
#include "portdnn/helpers/dims.h"

namespace sycldnn {
namespace gather {

/** Tensor sizes for a given Gather operation. */
struct GatherSizes {
  /** The type of the sizes is size_t since computed using
   * unsigned integer values from parameters.*/
  using Index = size_t;

  /** The size of the input tensor in elements. */
  Index input_size;

  /** The size of the output tensor in elements. */
  Index output_size;

  /** The size of the indices tensor in elements. */
  Index indices_size;

  /** The size of a gather block in elements, which counts the product
   * of innermost dimensions starting [axis + 1].
   */
  int block_size;

  /** The input dimension along axis provided as Gather parameter */
  int indices_max;
};

/**
 * Compute the block size & tensor sizes used in a Gather operator for the
 * specified parameters.
 * \param params The gather parameters containing the tensors dims and axis.
 * \return Returns a \ref sycldnn::gather::GatherSizes instance, containing
 *         the sizes of the tensors and the gather block in elements, and
 *         the max indice value along the gather axis.
 */
inline GatherSizes get_sizes(GatherParams const& params) {
  // Bring Gather axis value to the positive range if negative.
  // This assumes that params.axis is in [-rank, rank-1] inclusive.
  auto axis = params.axis < 0
                  ? params.axis + static_cast<int>(params.input_dims.size())
                  : params.axis;

  // Compute input size as product of input dimensions.
  auto input_size = sycldnn::helpers::get_total_size(params.input_dims);

  // Compute block size as product of input dimensions starting [axis + 1].
  auto block_size =
      std::accumulate(params.input_dims.begin() + axis + 1,
                      params.input_dims.end(), 1, std::multiplies<>{});

  // Compute blocks count as product of input dimensions up to [axis] excluded.
  auto blocks_count =
      std::accumulate(params.input_dims.begin(),
                      params.input_dims.begin() + axis, 1, std::multiplies<>{});

  // Compute indices size in elements as product of indices dimensions.
  auto indices_size = sycldnn::helpers::get_total_size(params.indices_dims);

  // Compute output size in elements as product of output dimensions.
  auto output_size = blocks_count * indices_size * block_size;

  // Max indices value as input dimension along the gather axis.
  auto indices_max = params.input_dims.at(axis);

  GatherSizes sizes{input_size, output_size, indices_size, block_size,
                    indices_max};

  return sizes;
}

}  // namespace gather
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_GATHER_SIZES_H_
