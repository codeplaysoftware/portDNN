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
#ifndef PORTDNN_INCLUDE_SCATTER_ND_PARAMS_H_
#define PORTDNN_INCLUDE_SCATTER_ND_PARAMS_H_

#include <vector>
#include "portdnn/data_format.h"

/**
 * \file
 * Contains the declaration of the \ref sycldnn::scatter_nd::ScatterNDParams
 * structure, which represents the tensor shapes and axis for a scatterND
 * operation.
 */
namespace sycldnn {
namespace scatter_nd {

/** Parameter struct containing the parameters required for a scatterND
 * operation.
 */
struct ScatterNDParams {
  /** The underlying data type of all index parameters. */
  using Index = int;
  /** List of input dimensions*/
  std::vector<Index> index_dims;
  /** List of index dimensions*/
  std::vector<Index> input_dims;
};

}  // namespace scatter_nd
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_SCATTER_ND_PARAMS_H_
