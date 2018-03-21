/*
 * Copyright 2018 Codeplay Software Ltd
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
#ifndef SYCLDNN_INCLUDE_CONV2D_SIZES_H_
#define SYCLDNN_INCLUDE_CONV2D_SIZES_H_

#include "sycldnn/conv2d/conv_type.h"
#include "sycldnn/conv2d/params.h"

namespace sycldnn {
namespace conv2d {
/** Tensor sizes for a given convolution. */
struct ConvSizes {
  size_t input_size;
  size_t filter_size;
  size_t output_size;
};
/**
 * Compute the total sizes of the tensors used in a convolution for the
 * specified parameters.
 */
template <typename ConvType>
ConvSizes get_sizes(Conv2DParams const& params);
template <>
inline ConvSizes get_sizes<conv_type::Forward>(Conv2DParams const& params) {
  size_t inp_size =
      params.batch * params.in_rows * params.in_cols * params.channels;
  size_t fil_size = params.window_rows * params.window_cols * params.channels *
                    params.features;
  size_t out_size =
      params.batch * params.out_rows * params.out_cols * params.features;
  ConvSizes sizes{inp_size, fil_size, out_size};
  return sizes;
}
template <>
inline ConvSizes get_sizes<conv_type::InputBackprop>(
    Conv2DParams const& params) {
  size_t inp_size =
      params.batch * params.out_rows * params.out_cols * params.features;
  size_t fil_size = params.window_rows * params.window_cols * params.channels *
                    params.features;
  size_t out_size =
      params.batch * params.in_rows * params.in_cols * params.channels;
  ConvSizes sizes{inp_size, fil_size, out_size};
  return sizes;
}
template <>
inline ConvSizes get_sizes<conv_type::FilterBackprop>(
    Conv2DParams const& params) {
  size_t inp_size =
      params.batch * params.in_rows * params.in_cols * params.channels;
  size_t fil_size =
      params.batch * params.out_rows * params.out_cols * params.features;
  size_t out_size = params.window_rows * params.window_cols * params.channels *
                    params.features;
  ConvSizes sizes{inp_size, fil_size, out_size};
  return sizes;
}
}  // namespace conv2d
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_CONV2D_SIZES_H_
