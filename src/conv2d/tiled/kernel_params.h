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
#ifndef PORTDNN_SRC_CONV2D_TILED_KERNEL_PARAMS_H_
#define PORTDNN_SRC_CONV2D_TILED_KERNEL_PARAMS_H_

#include "portdnn/conv2d/params.h"

namespace sycldnn {
namespace conv2d {
namespace internal {

template <typename ConvType>
inline Conv2DParams get_kernel_params(Conv2DParams params) {
  return params;
}
template <>
inline Conv2DParams get_kernel_params<conv_type::InputBackprop>(
    Conv2DParams params) {
  // We need to change the padding from input padding to output padding for
  // the kernel. pad_out = filt_size - 1 - pad_in
  params.pad_rows = params.window_rows - 1 - params.pad_rows;
  params.pad_cols = params.window_cols - 1 - params.pad_cols;
  return params;
}
template <>
inline Conv2DParams get_kernel_params<conv_type::FilterBackprop>(
    Conv2DParams params) {
  // Map the input dimensions to those expected in the convolution kernel.
  const auto window_rows =
      params.out_rows * params.stride_rows - (params.stride_rows - 1);
  const auto window_cols =
      params.out_cols * params.stride_cols - (params.stride_cols - 1);
  params.out_rows = params.window_rows;
  params.out_cols = params.window_cols;
  params.window_rows = window_rows;
  params.window_cols = window_cols;
  return params;
}

}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_SRC_CONV2D_TILED_KERNEL_PARAMS_H_
