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
#ifndef PORTDNN_INCLUDE_DEPTHWISE_CONV2D_SIZES_H_
#define PORTDNN_INCLUDE_DEPTHWISE_CONV2D_SIZES_H_

/**
 * \file
 * Contains functionality for calculating the size of tensors from the
 * convolution parameters, including the declaration of the
 * \ref sycldnn::depthwise_conv2d::ConvSizes structure.
 */
#include "portdnn/conv2d/conv_type.h"
#include "portdnn/depthwise_conv2d/params.h"

namespace sycldnn {
namespace depthwise_conv2d {

/** Tensor sizes for a given convolution. */
struct ConvSizes {
  /** The size of the input tensor in elements. */
  size_t input_size;
  /** The size of the filter tensor in elements. */
  size_t filter_size;
  /** The size of the output tensor in elements. */
  size_t output_size;
};

/**
 * Compute the total sizes of the tensors used in a depthwise convolution for
 * the specified parameters.
 * \param params The convolution parameters, containing the tensor sizes and
 *               filter strides.
 * \return Returns a \ref sycldnn::depthwise_conv2d::ConvSizes instance,
 *         containing the sizes of the tensors in elements.
 */
template <typename ConvType>
ConvSizes get_sizes(DepthwiseConv2DParams const& params);

/** \copydoc sycldnn::conv2d::get_sizes(Conv2DParams const& params) */
template <>
inline ConvSizes get_sizes<conv2d::conv_type::Forward>(
    DepthwiseConv2DParams const& params) {
  size_t inp_size =
      params.batch * params.in_rows * params.in_cols * params.channels;
  size_t fil_size = params.window_rows * params.window_cols * params.channels *
                    params.channel_multiplier;
  size_t out_size = params.batch * params.out_rows * params.out_cols *
                    params.channels * params.channel_multiplier;

  return ConvSizes{inp_size, fil_size, out_size};
}

/** \copydoc sycldnn::conv2d::get_sizes(Conv2DParams const& params) */
template <>
inline ConvSizes get_sizes<conv2d::conv_type::InputBackprop>(
    DepthwiseConv2DParams const& params) {
  size_t inp_size = params.batch * params.out_rows * params.out_cols *
                    params.channels * params.channel_multiplier;
  size_t fil_size = params.window_rows * params.window_cols * params.channels *
                    params.channel_multiplier;
  size_t out_size =
      params.batch * params.in_rows * params.in_cols * params.channels;

  return ConvSizes{inp_size, fil_size, out_size};
}

/** \copydoc sycldnn::conv2d::get_sizes(Conv2DParams const& params) */
template <>
inline ConvSizes get_sizes<conv2d::conv_type::FilterBackprop>(
    DepthwiseConv2DParams const& params) {
  size_t inp_size =
      params.batch * params.in_rows * params.in_cols * params.channels;
  size_t fil_size = params.batch * params.out_rows * params.out_cols *
                    params.channels * params.channel_multiplier;
  size_t out_size = params.window_rows * params.window_cols * params.channels *
                    params.channel_multiplier;

  return ConvSizes{inp_size, fil_size, out_size};
}

}  // namespace depthwise_conv2d
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_DEPTHWISE_CONV2D_SIZES_H_
