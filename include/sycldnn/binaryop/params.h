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
#ifndef SYCLDNN_INCLUDE_BINARYOP_PARAMS_H_
#define SYCLDNN_INCLUDE_BINARYOP_PARAMS_H_

#include "sycldnn/data_format.h"

/**
 * \file
 * Defines the \ref sycldnn::binaryop::BinaryParams struct,
 * which contains the values used in a binary operation.
 */
namespace sycldnn {
namespace binaryop {

/** Struct that contains values used in a Binary op. */
struct BinaryParams {
  /** The type of the params is int, providing a decent
   * upper bound on the tensor sizes.*/
  using Index = int;

  /** The number of items in input1. */
  Index lhs_items;

  /** The number of items in input2. */
  Index rhs_items;

  /** The data format used in the input and output tensors. */
  sycldnn::DataFormat input_format = sycldnn::DataFormat::NHWC;
};

}  // namespace binaryop
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_BINARYOP_PARAMS_H_
