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
#ifndef PORTDNN_INCLUDE_BINARYOP_PARAMS_H_
#define PORTDNN_INCLUDE_BINARYOP_PARAMS_H_

#include <vector>

#include "portdnn/data_format.h"

/**
 * \file
 * Defines the \ref sycldnn::binaryop::BinaryParams struct,
 * which contains the values used in a binary operation.
 */
namespace sycldnn {
namespace binaryop {

static constexpr int MAX_DIMS = 4;

/** Struct that contains values used in a Binary op. */
struct BinaryParams {
  /** The type of the params is int, providing a decent
   * upper bound on the tensor sizes.*/
  using Index = int;

  /** Left operand dimensions. */
  std::vector<Index> lhs_dims;

  /** Right operand dimensions. */
  std::vector<Index> rhs_dims;
};

}  // namespace binaryop
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_BINARYOP_PARAMS_H_
