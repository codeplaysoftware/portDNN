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
#ifndef PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_KERNEL_PARAMS_H_
#define PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_KERNEL_PARAMS_H_

#include "portdnn/conv2d/params.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace im2col {

/**
 * Get the conv2d parameters as expected by the im2col kernels.
 *
 * \param params Initial user provided parameters
 * \return Parameters expected by the kernels
 */
template <typename ConvType>
inline Conv2DParams get_kernel_params(Conv2DParams params) {
  return params;
}

template <>
inline Conv2DParams get_kernel_params<conv_type::FilterBackprop>(
    Conv2DParams params) {
  std::swap(params.out_rows, params.window_rows);
  std::swap(params.out_cols, params.window_cols);
  std::swap(params.stride_rows, params.dilation_rows);
  std::swap(params.stride_cols, params.dilation_cols);
  return params;
}

}  // namespace im2col
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_KERNEL_PARAMS_H_
