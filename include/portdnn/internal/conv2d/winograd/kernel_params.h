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
#ifndef PORTDNN_INCLUDE_INTERNAL_CONV2D_WINOGRAD_KERNEL_PARAMS_H_
#define PORTDNN_INCLUDE_INTERNAL_CONV2D_WINOGRAD_KERNEL_PARAMS_H_

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/params.h"

/**
 * \file
 * Contains the sycldnn::conv2d::internal::winograd::get_params() helper
 * function to convert user provided Conv2DParams into the parameters expected
 * by the Winograd kernels.
 */

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace winograd {

/**
 * Convert the user provided params to those expected by the kernels.
 * \param params User parameters to convert
 * \return Parameters expected by the Winograd kernels
 */
template <typename ConvType>
inline Conv2DParams get_params(Conv2DParams params) {
  return params;
}

/** \copydoc get_params() */
template <>
inline Conv2DParams get_params<conv_type::InputBackprop>(Conv2DParams params) {
  std::swap(params.channels, params.features);
  std::swap(params.in_rows, params.out_rows);
  std::swap(params.in_cols, params.out_cols);
  // We need to change the padding from input padding to output padding for
  // the winograd matmul kernel. pad_out = filt_size - 1 - pad_in
  params.pad_rows = params.window_rows - 1 - params.pad_rows;
  params.pad_cols = params.window_cols - 1 - params.pad_cols;
  return params;
}

/** \copydoc get_params() */
template <>
inline Conv2DParams get_params<conv_type::FilterBackprop>(Conv2DParams params) {
  std::swap(params.out_rows, params.window_rows);
  std::swap(params.out_cols, params.window_cols);
  return params;
}

}  // namespace winograd
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_INTERNAL_CONV2D_WINOGRAD_KERNEL_PARAMS_H_
