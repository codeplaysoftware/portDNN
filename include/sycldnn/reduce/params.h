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
#ifndef SYCLDNN_INCLUDE_REDUCE_PARAMS_H_
#define SYCLDNN_INCLUDE_REDUCE_PARAMS_H_

#include <vector>

/**
 * \file
 * Defines the \ref sycldnn::reduce::NDReduceParams struct,
 * which contains the values used in a nd reduce operation.
 */
namespace sycldnn {
namespace reduce {
/** Struct that contains values used in a ND reduction op. */
struct NDReduceParams {
  /** The type of the params is int, providing a decent
   * upper bound on the tensor sizes.*/
  using Index = int;

  /** Dimensions of input of reduction */
  std::vector<Index> dims;

  /** Vector of indices of which dimensions of the tensor are to be reduced*/
  std::vector<Index> axes;

  /** Boolean to be set true if the output tensor needs to be memset to
   * default values before the reduce operation. (Result may not be correct
   * this is set to false and and output is not set to default value for
   * operation)*/
  bool prepare_output = true;
};

}  // namespace reduce
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_REDUCE_PARAMS_H_
