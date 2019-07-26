
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
#ifndef SYCLDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_LAUNCH_INPUT_TRANSFORM_H_
#define SYCLDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_LAUNCH_INPUT_TRANSFORM_H_

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/conv2d/params.h"
#include "sycldnn/conv2d/sizes.h"

#include "sycldnn/internal/conv2d/im2col/full_pointer_set.h"
#include "sycldnn/internal/conv2d/im2col/tile_info.h"

#include "sycldnn/export.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace im2col {

/**
 * Launch the input transform to expand the input tensor images.
 *
 * Implemented in the compiled SYCL-DNN library.
 *
 * \param [in]  input     User provided input tensor
 * \param [out] output    Input transform tensor to fill with transformed values
 * \param [in]  params    Kernel parameters for the convolution
 * \param [in]  queue     SYCL queue to enqueue the kernel to
 * \param [in]  n_tiles   Total number of im2col tiles in transform
 * \param [in]  tile_size Number of elements in each im2col tile
 * \return An SNNStatus with event linked to the kernel launch or an error code.
 */
template <typename T, typename ConvType>
SNN_EXPORT SNNStatus launch_input_transform(BaseMemObject<T const>& input,
                                            BaseMemObject<T>& output,
                                            Conv2DParams const& params,
                                            int n_tiles, int tile_size,
                                            cl::sycl::queue& queue);

/** Extract the buffers from the backend and call the kernel launcher. */
template <typename T, typename ConvType, typename Backend>
static SNNStatus launch_input_transform(
    FullPointerSet<T, Backend, ConvType> const& pointers, size_t in_offset,
    TileInfo const& tile_info, Conv2DParams const& params, Backend& backend) {
  auto const conv_sizes = get_sizes<ConvType>(params);
  size_t const input_size = conv_sizes.input_size;
  auto input_acc =
      backend.get_mem_object_internal(pointers.input + in_offset, input_size);

  int n_tiles;
  int tile_size;
  if (std::is_same<ConvType, conv_type::FilterBackprop>::value) {
    n_tiles = tile_info.number;
    tile_size = params.batch * tile_info.size;
  } else {
    n_tiles = params.batch * tile_info.number;
    tile_size = tile_info.size;
  }
  size_t const transform_size = n_tiles * tile_size;
  auto transform_acc =
      backend.get_mem_object_internal(pointers.transform, transform_size);

  cl::sycl::queue queue = backend.get_queue();
  return launch_input_transform<T, ConvType>(input_acc, transform_acc, params,
                                             n_tiles, tile_size, queue);
}

}  // namespace im2col
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_LAUNCH_INPUT_TRANSFORM_H_
