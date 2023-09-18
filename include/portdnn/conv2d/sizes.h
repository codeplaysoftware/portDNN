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
#ifndef PORTDNN_INCLUDE_CONV2D_SIZES_H_
#define PORTDNN_INCLUDE_CONV2D_SIZES_H_

/**
 * \file
 * Contains functionality for calculating the size of tensors from the
 * convolution parameters, including the declaration of the
 * \ref sycldnn::conv2d::ConvSizes structure.
 */
#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/params.h"

#include <cstddef>

namespace sycldnn {
namespace conv2d {

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
 * \brief Compute the batch of the tensors used in a convolution for the
 * specified parameters.
 *
 * \param params The convolution parameters, containing the tensor sizes and
 *               filter strides.
 * \return Returns a \ref sycldnn::conv2d::ConvSizes instance, containing the
 *         sizes of the tensors in elements.
 */
template <typename ConvType>
inline ConvSizes get_batch_sizes(Conv2DParams const& params) {
  size_t batch = params.batch;
  ConvSizes sizes{batch, 1, batch};
  return sizes;
}

/** \copydoc sycldnn::conv2d::get_batch_sizes(Conv2DParams const& params) */
template <>
inline ConvSizes get_batch_sizes<conv_type::FilterBackprop>(
    Conv2DParams const& params) {
  size_t batch = params.batch;
  ConvSizes sizes{batch, batch, 1};
  return sizes;
}

/**
 * \brief Compute the spatial sizes (height * width) of the tensors used in a
 * convolution for the specified parameters.
 *
 * \param params The convolution parameters, containing the tensor sizes and
 *               filter strides.
 * \return Returns a \ref sycldnn::conv2d::ConvSizes instance, containing the
 *         sizes of the tensors in elements.
 */
template <typename ConvType>
ConvSizes get_spatial_sizes(Conv2DParams const& params);

/** \copydoc sycldnn::conv2d::get_spatial_sizes(Conv2DParams const& params) */
template <>
inline ConvSizes get_spatial_sizes<conv_type::Forward>(
    Conv2DParams const& params) {
  size_t inp_size = params.in_rows * params.in_cols;
  size_t fil_size = params.window_rows * params.window_cols;
  size_t out_size = params.out_rows * params.out_cols;
  ConvSizes sizes{inp_size, fil_size, out_size};
  return sizes;
}

/** \copydoc sycldnn::conv2d::get_spatial_sizes(Conv2DParams const& params) */
template <>
inline ConvSizes get_spatial_sizes<conv_type::InputBackprop>(
    Conv2DParams const& params) {
  size_t inp_size = params.out_rows * params.out_cols;
  size_t fil_size = params.window_rows * params.window_cols;
  size_t out_size = params.in_rows * params.in_cols;
  ConvSizes sizes{inp_size, fil_size, out_size};
  return sizes;
}

/** \copydoc sycldnn::conv2d::get_spatial_sizes(Conv2DParams const& params) */
template <>
inline ConvSizes get_spatial_sizes<conv_type::FilterBackprop>(
    Conv2DParams const& params) {
  size_t inp_size = params.in_rows * params.in_cols;
  size_t fil_size = params.out_rows * params.out_cols;
  size_t out_size = params.window_rows * params.window_cols;
  ConvSizes sizes{inp_size, fil_size, out_size};
  return sizes;
}

/**
 * \brief Compute the spatial sizes (channel and/or feature) of the tensors used
 * in a convolution for the specified parameters.
 *
 * \param params The convolution parameters, containing the tensor sizes and
 *               filter strides.
 * \return Returns a \ref sycldnn::conv2d::ConvSizes instance, containing the
 *         sizes of the tensors in elements.
 */
template <typename ConvType>
ConvSizes get_channel_sizes(Conv2DParams const& params);

/** \copydoc sycldnn::conv2d::get_channel_sizes(Conv2DParams const& params) */
template <>
inline ConvSizes get_channel_sizes<conv_type::Forward>(
    Conv2DParams const& params) {
  size_t inp_size = params.channels;
  size_t fil_size = params.channels * params.features / params.groups;
  size_t out_size = params.features / params.groups;
  ConvSizes sizes{inp_size, fil_size, out_size};
  return sizes;
}

/** \copydoc sycldnn::conv2d::get_channel_sizes(Conv2DParams const& params) */
template <>
inline ConvSizes get_channel_sizes<conv_type::InputBackprop>(
    Conv2DParams const& params) {
  size_t inp_size = params.features;
  size_t fil_size = params.channels * params.features;
  size_t out_size = params.channels;
  ConvSizes sizes{inp_size, fil_size, out_size};
  return sizes;
}

/** \copydoc sycldnn::conv2d::get_channel_sizes(Conv2DParams const& params) */
template <>
inline ConvSizes get_channel_sizes<conv_type::FilterBackprop>(
    Conv2DParams const& params) {
  size_t inp_size = params.channels;
  size_t fil_size = params.features;
  size_t out_size = params.channels * params.features;
  ConvSizes sizes{inp_size, fil_size, out_size};
  return sizes;
}

/**
 * Compute the total sizes of the tensors used in a convolution for the
 * specified parameters.
 * \param params The convolution parameters, containing the tensor sizes and
 *               filter strides.
 * \return Returns a \ref sycldnn::conv2d::ConvSizes instance, containing the
 *         sizes of the tensors in elements.
 */
template <typename ConvType>
inline ConvSizes get_sizes(Conv2DParams const& params) {
  ConvSizes batch_sizes = get_batch_sizes<ConvType>(params);
  ConvSizes spatial_sizes = get_spatial_sizes<ConvType>(params);
  ConvSizes channel_sizes = get_channel_sizes<ConvType>(params);
  size_t inp_size = batch_sizes.input_size * spatial_sizes.input_size *
                    channel_sizes.input_size;
  size_t fil_size = batch_sizes.filter_size * spatial_sizes.filter_size *
                    channel_sizes.filter_size;
  size_t out_size = batch_sizes.output_size * spatial_sizes.output_size *
                    channel_sizes.output_size * params.groups;
  ConvSizes sizes{inp_size, fil_size, out_size};
  return sizes;
}
}  // namespace conv2d
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_CONV2D_SIZES_H_
