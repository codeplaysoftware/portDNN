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
#ifndef PORTDNN_INCLUDE_SCATTER_ND_SIZES_H_
#define PORTDNN_INCLUDE_SCATTER_ND_SIZES_H_

/**
 * \file
 * Contains functionality for calculating the tensor size, slice size, index
 * depth as well as index offsets from the ScatterND parameters, including the
 * declaration of the \ref sycldnn::scatter_nd::ScatterNDSizes structure.
 */
#include <functional>
#include <numeric>

#include "portdnn/scatter_nd/params.h"

namespace sycldnn {
namespace scatter_nd {

/** Tensor sizes for a ScatterND operation. */
struct ScatterNDSizes {
  /** Index type*/
  using Index = int;

  /** The number of dimensions in the input tensor*/
  Index rank;

  /** The number of updates to be applied to the output tensor. (First dimension
   * of the index and update tensors.) */
  Index num_updates;

  /** The rank of each index. When it is equal to the rank of the input tensor
   * then and elementwise update is done, else a slice is updated. (Second
   * dimension of the index tensor.) */
  Index index_depth;

  /** The size of each update being made. (Second dimension of the update
   * tensor). */
  Index slice_size;

  /** The size of the input tensor which is equal to the size of the output
   * tensor*/
  Index output_size;

  /** First dimension of input/output tensor*/
  Index dim_0;
  /** Second dimension of input/output tensor*/
  Index dim_1;
  /** Third dimension of input/output tensor*/
  Index dim_2;
  /** Fourth dimension of input/output tensor*/
  Index dim_3;
};

/**
 * Compute the slice size used in a ScatterND operator for the
 * specified parameters.
 * \param params The scatterND parameters containing the tensors dims and
 * index/update dims. \return Returns a \ref sycldnn::scatter_nd::ScatterNDSizes
 * instance, containing the tensor size, slice size, index depth as well as
 * index offsets
 */
inline ScatterNDSizes get_sizes(ScatterNDParams const& params) {
  using Index = int;

  Index num_updates = params.index_dims[0];
  Index index_depth = params.index_dims[1];
  Index rank = params.input_dims.size();
  Index slice_size =
      std::accumulate(params.input_dims.begin() + index_depth,
                      params.input_dims.end(), 1, std::multiplies<int>());

  auto dim_0 = params.input_dims[0];
  auto dim_1 = rank > 1 ? params.input_dims[1] : 1;
  auto dim_2 = rank > 2 ? params.input_dims[2] : 1;
  auto dim_3 = rank > 3 ? params.input_dims[3] : 1;

  ScatterNDSizes sizes{rank,
                       num_updates,
                       index_depth,
                       slice_size,
                       dim_0 * dim_1 * dim_2 * dim_3,
                       dim_0,
                       dim_1,
                       dim_2,
                       dim_3};
  return sizes;
}

}  // namespace scatter_nd
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_SCATTER_ND_SIZES_H_
