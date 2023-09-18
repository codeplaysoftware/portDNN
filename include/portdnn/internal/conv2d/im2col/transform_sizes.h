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

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/params.h"

#include "portdnn/internal/conv2d/im2col/tile_info.h"

#ifndef PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_TRANSFORM_SIZES_H_
#define PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_TRANSFORM_SIZES_H_
namespace sycldnn {
namespace conv2d {
namespace internal {
namespace im2col {
/** Struct containing the tensor transform info*/
struct ConvTransformSizes {
  size_t filter_transform_size;
  size_t output_transform_size;
  size_t input_transform_size;

  size_t get_transform_offset() const {
    return filter_transform_size + output_transform_size;
  }
};

/** Get the tensor size needed for the filter transform. */
template <typename ConvType>
size_t filter_transform_size(Conv2DParams const& params) {
  if ((params.groups == 1 ||
       (params.group_format == sycldnn::BatchFormat::INTERLEAVED &&
        params.filter_format == sycldnn::FilterFormat::HWCF) ||
       (params.group_format == sycldnn::BatchFormat::STRIDED &&
        params.filter_format == sycldnn::FilterFormat::FHWC))) {
    return 0;
  }
  return params.window_rows * params.window_cols * params.channels *
         params.features / params.groups;
}

template <>
size_t filter_transform_size<conv_type::InputBackprop>(
    Conv2DParams const& params) {
  return params.window_rows * params.window_cols * params.channels *
         params.features / params.groups;
}

/** Get the tensor size needed for the output transform. */
template <typename ConvType>
size_t output_transform_size(Conv2DParams const& params) {
  if (!std::is_same<ConvType, conv_type::Forward>::value ||
      params.groups == 1 ||
      (params.group_format == sycldnn::BatchFormat::INTERLEAVED &&
       params.filter_format == sycldnn::FilterFormat::HWCF)) {
    return 0;
  }
  return params.out_rows * params.out_cols * params.features;
}

/** Get the tensor size needed for the input transform. */
template <typename ConvType>
size_t input_transform_size(Conv2DParams const& params) {
  auto const tile_info = im2col::get_tile_info<ConvType>(params);
  return params.groups * tile_info.number * tile_info.size;
}

template <typename ConvType>
ConvTransformSizes get_transform_sizes(Conv2DParams const& params) {
  ConvTransformSizes transform_sizes;
  transform_sizes.input_transform_size = input_transform_size<ConvType>(params);
  transform_sizes.filter_transform_size =
      filter_transform_size<ConvType>(params);
  transform_sizes.output_transform_size =
      output_transform_size<ConvType>(params);
  return transform_sizes;
}

}  // namespace im2col
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn
#endif
