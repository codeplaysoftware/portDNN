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
#ifndef PORTDNN_SRC_DEPTHWISE_CONV2D_KERNEL_PARAMS_H_
#define PORTDNN_SRC_DEPTHWISE_CONV2D_KERNEL_PARAMS_H_

#include <algorithm>

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/depthwise_conv2d/params.h"

namespace sycldnn {
namespace depthwise_conv2d {
namespace internal {

template <typename ConvType>
inline DepthwiseConv2DParams get_kernel_params(DepthwiseConv2DParams params) {
  return params;
}

template <>
inline DepthwiseConv2DParams
get_kernel_params<conv2d::conv_type::InputBackprop>(
    DepthwiseConv2DParams params) {
  params.pad_rows = params.window_rows - params.pad_rows - 1;
  params.pad_cols = params.window_cols - params.pad_cols - 1;
  return params;
}

template <>
inline DepthwiseConv2DParams
get_kernel_params<conv2d::conv_type::FilterBackprop>(
    DepthwiseConv2DParams params) {
  std::swap(params.out_rows, params.window_rows);
  std::swap(params.out_cols, params.window_cols);
  return params;
}

}  // namespace internal
}  // namespace depthwise_conv2d
}  // namespace sycldnn

#endif  // PORTDNN_SRC_DEPTHWISE_CONV2D_KERNEL_PARAMS_H_
