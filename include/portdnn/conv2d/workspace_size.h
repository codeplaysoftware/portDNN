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
#ifndef PORTDNN_INCLUDE_CONV2D_WORKSPACE_SIZE_H_
#define PORTDNN_INCLUDE_CONV2D_WORKSPACE_SIZE_H_

#include "portdnn/conv2d/algorithm.h"
#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/selector/selector.h"

#include "portdnn/internal/conv2d/im2col/kernel_params.h"
#include "portdnn/internal/conv2d/im2col/tile_info.h"
#include "portdnn/internal/conv2d/im2col/transform_sizes.h"

#include "portdnn/internal/conv2d/winograd/kernel_params.h"
#include "portdnn/internal/conv2d/winograd/tile_info.h"

namespace sycldnn {
namespace conv2d {

/**
 * Sizes required for a user provided workspace buffer.
 *
 * Both a minimum required size and a recommmended size are provided, so that on
 * low memory systems a user can choose to possibly sacrifice performance for
 * less memory usage. If a workspace smaller than the recommended size is used
 * then the work will be batched into a number of kernels, rather than run in
 * one.
 */
struct WorkspaceSize {
  /** Minimum number of elements that a workspace buffer must hold. */
  size_t required_size;
  /** Recommended number of elements that a workspace buffer should hold. */
  size_t recommended_size;
};

namespace internal {

/** Get the workspace sizes for Winograd using the tile sizes specified in the
 * template parameters. */
template <typename ConvType, int M, int N, int R, int S>
WorkspaceSize winograd_impl_workspace_size(Conv2DParams const& params) {
  static constexpr int A = M + R - 1;
  static constexpr int B = N + S - 1;
  auto kernel_params = winograd::get_params<ConvType>(params);
  auto const tile_info =
      winograd::get_tile_info<ConvType, M, N, R, S>(kernel_params);

  size_t input_transform_size =
      A * B * tile_info.number * kernel_params.channels;
  size_t inter_transform_size =
      A * B * tile_info.number * kernel_params.features;
  size_t filter_transform_size =
      A * B * kernel_params.channels * kernel_params.features;
  size_t required_size =
      input_transform_size + inter_transform_size + filter_transform_size;
  size_t recommended_size =
      params.batch * (input_transform_size + inter_transform_size) +
      filter_transform_size;
  return {required_size, recommended_size};
}

/** Get the workspace sizes for Winograd using the smaller tile sizes. */
template <typename ConvType>
WorkspaceSize workspace_size_for_winograd(Conv2DParams const& params) {
  // The choice of tile sizes here should match that used in
  // src/conv2d/winoograd/launch.cc
  if (std::is_same<ConvType, conv_type::FilterBackprop>::value) {
    return winograd_impl_workspace_size<ConvType, 3, 3, 2, 2>(params);
  } else {
    return winograd_impl_workspace_size<ConvType, 2, 2, 3, 3>(params);
  }
}

/** Get the workspace sizes for Winograd using the larger tile sizes. */
template <typename ConvType>
WorkspaceSize workspace_size_for_winograd_large(Conv2DParams const& params) {
  // The choice of tile sizes here should match that used in
  // src/conv2d/winoograd/launch.cc
  if (std::is_same<ConvType, conv_type::FilterBackprop>::value) {
    return winograd_impl_workspace_size<ConvType, 3, 3, 3, 3>(params);
  } else {
    return winograd_impl_workspace_size<ConvType, 4, 4, 3, 3>(params);
  }
}

/** Get the workspace sizes needed for the Im2col transform tensors. */
template <typename ConvType>
WorkspaceSize workspace_size_for_im2col(Conv2DParams const& params) {
  auto const transform_sizes = im2col::get_transform_sizes<ConvType>(params);

  size_t required_size;
  size_t recommended_size;
  // im2col convolution needs a workspace buffer large enough to hold
  // the input transform and the filter transform tensors for one image.
  required_size = transform_sizes.input_transform_size +
                  transform_sizes.filter_transform_size;
  recommended_size = (params.batch * transform_sizes.input_transform_size) +
                     transform_sizes.filter_transform_size;
  if (params.groups > 1 &&
      params.group_format == sycldnn::BatchFormat::STRIDED) {
    // NHWC strided group convolution also requires memory in the
    // workspace buffer large enough to transpose the output result
    required_size += transform_sizes.output_transform_size;
    recommended_size += params.batch * transform_sizes.output_transform_size;
  }
  return {required_size, recommended_size};
}

/** Get the WorkspaceSize for the specified convolution using the provided
 * Algorithm. */
template <typename ConvType>
WorkspaceSize query_workspace_size(Conv2DParams const& params,
                                   Algorithm algorithm) {
  switch (algorithm) {
    case Algorithm::Winograd:
      return workspace_size_for_winograd<ConvType>(params);
      break;
    case Algorithm::WinogradLarge:
      return workspace_size_for_winograd_large<ConvType>(params);
      break;
    case Algorithm::Im2col:
      return workspace_size_for_im2col<ConvType>(params);
      break;
    case Algorithm::Direct:
    case Algorithm::Tiled:
    case Algorithm::Matmul:
    case Algorithm::NotSupported:
      return {0, 0};
  }
  SNN_ASSERT(false, "Invalid algorithm passed to query_workspace_size.");
  return {0, 0};
}
}  // namespace internal

/**
 * Query the number of elements that a workspace buffer must hold in order to be
 * used in a convolution computation.
 *
 * \param params Convolution parameters describing the computation.
 * \param selector Selector to use to determine which algorithm to use.
 *
 * \return A WorkspaceSize struct containing the minimum required and
 *         recommended number of elements that a workspace buffer should hold.
 */
template <typename ConvType>
WorkspaceSize query_workspace_size(Conv2DParams const& params,
                                   Selector& selector) {
  return internal::query_workspace_size<ConvType>(
      params, selector.select<ConvType>(params));
}

}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_CONV2D_WORKSPACE_SIZE_H_
