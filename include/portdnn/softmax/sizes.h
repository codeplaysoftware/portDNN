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
#ifndef PORTDNN_INCLUDE_SOFTMAX_SIZES_H_
#define PORTDNN_INCLUDE_SOFTMAX_SIZES_H_

/**
 * \file
 * Contains functionality for calculating the size of tensors from the
 * Softmax parameters, including the declaration of the
 * \ref sycldnn::softmax::SoftmaxSizes structure.
 */
#include "portdnn/softmax/params.h"

namespace sycldnn {
namespace softmax {

/** Tensor sizes for a given Softmax operation. */
struct SoftmaxSizes {
  /** The size of the input tensor in elements. */
  int input_size;
  /** The size of the workspace tensor in elements. */
  int workspace_size;
  /** The size of the output tensor in elements. */
  int output_size;
};

/**
 * Compute the total sizes of the tensors used in a Softmax operator for the
 * specified parameters.
 * \param params The softmax parameters containing the tensor sizes.
 * \return Returns a \ref sycldnn::softmax::SoftmaxSizes instance, containing
 *         the sizes of the tensors in elements.
 */
inline SoftmaxSizes get_sizes(SoftmaxParams const& params) {
  int input = params.batch * params.rows * params.cols * params.channels;

  int workspace = params.batch * params.rows * params.cols;

  SoftmaxSizes sizes{input, workspace, input};
  return sizes;
}

}  // namespace softmax
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_SOFTMAX_SIZES_H_
