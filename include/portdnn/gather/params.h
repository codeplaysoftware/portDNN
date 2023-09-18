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
 * distributed under the License is distributed on an "AS IS" BASIS
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef PORTDNN_INCLUDE_GATHER_PARAMS_H_
#define PORTDNN_INCLUDE_GATHER_PARAMS_H_

#include <cstdint>
#include <vector>

/**
 * \file
 * Defines the \ref sycldnn::gather::GatherParams struct,
 * which contains the values used in a gather operation.
 */
namespace sycldnn {
namespace gather {

/** Struct that contains values used in a Gather op.
 */
struct GatherParams {
  /** The underlying data type of all index parameters. */
  using Index = int;

  /** The input dimensions */
  std::vector<Index> input_dims;

  /** The indices dimensions */
  std::vector<Index> indices_dims;

  /** The input axis on which gather is applied */
  int axis;
};

}  // namespace gather
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_GATHER_PARAMS_H_
