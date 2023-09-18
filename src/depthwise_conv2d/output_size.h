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
#ifndef PORTDNN_SRC_DEPTHWISE_CONV2D_OUTPUT_SIZE_H_
#define PORTDNN_SRC_DEPTHWISE_CONV2D_OUTPUT_SIZE_H_

#include "portdnn/conv2d/conv_type.h"

#include "portdnn/helpers/minmax.h"
#include "portdnn/helpers/ratio.h"

#include "portdnn/depthwise_conv2d/params.h"

#include <cstddef>

namespace sycldnn {
namespace depthwise_conv2d {
namespace internal {

template <typename ConvType>
inline size_t get_output_size(DepthwiseConv2DParams const& params);

template <>
inline size_t get_output_size<conv2d::conv_type::Forward>(
    DepthwiseConv2DParams const& params) {
  return params.batch * params.out_rows * params.out_cols * params.channels *
         params.channel_multiplier;
}
template <>
inline size_t get_output_size<conv2d::conv_type::InputBackprop>(
    DepthwiseConv2DParams const& params) {
  return params.batch * params.in_rows * params.in_cols * params.channels;
}
template <>
inline size_t get_output_size<conv2d::conv_type::FilterBackprop>(
    DepthwiseConv2DParams const& params) {
  return params.window_rows * params.window_cols * params.channels *
         params.channel_multiplier;
}

}  // namespace internal
}  // namespace depthwise_conv2d
}  // namespace sycldnn

#endif  // PORTDNN_SRC_DEPTHWISE_CONV2D_OUTPUT_SIZE_H_
