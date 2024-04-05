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
#ifndef SYCLDNN_INCLUDE_CONCAT_PARAMS_H_
#define SYCLDNN_INCLUDE_CONCAT_PARAMS_H_

#include "sycldnn/data_format.h"

/**
 * \file
 * Defines the \ref sycldnn::concat::ConcatParams struct,
 * which contains the values used in a concat operation.
 */
namespace sycldnn {
namespace concat {

/** Struct that contains values used in a concat op. */
struct ConcatParams {
  /** Vector of dimensions for each input tensor */
  std::vector<std::vector<int>> input_descriptors;
  /** Rank of input tensors */
  int rank = 0;
  /** The axis along which we are concatenating */
  int axis = 0;
};

}  // namespace concat
}  // namespace sycldnn

#endif