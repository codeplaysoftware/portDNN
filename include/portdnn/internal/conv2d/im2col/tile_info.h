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
#ifndef PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_TILE_INFO_H_
#define PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_TILE_INFO_H_

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/params.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace im2col {

/** Information about the im2col tiles for a given convolution. */
struct TileInfo {
  /** Total number of tiles required for a single image. */
  int number;
  /** Size of a single tile. */
  int size;
};

/**
 * Get info about the tile sizes used by im2col for a single image.
 *
 * Im2col transforms each input window into a 1D vector, which we refer to as a
 * tile. These tiles then make up one of the matrices which are used in the
 * im2col matrix multiply.
 *
 * \param params User provided conv2d parameters
 * \return A TileInfo struct containing the number of size of im2col tiles
 */
template <typename ConvType>
inline TileInfo get_tile_info(Conv2DParams const& params);
template <>
inline TileInfo get_tile_info<conv_type::Forward>(Conv2DParams const& params) {
  const int n_tiles = params.out_rows * params.out_cols;
  const int tile_size =
      params.window_rows * params.window_cols * params.channels / params.groups;
  return TileInfo{n_tiles, tile_size};
}
template <>
inline TileInfo get_tile_info<conv_type::InputBackprop>(
    Conv2DParams const& params) {
  const int n_tiles = params.in_rows * params.in_cols;
  const int tile_size =
      params.window_rows * params.window_cols * params.features;
  return TileInfo{n_tiles, tile_size};
}
template <>
inline TileInfo get_tile_info<conv_type::FilterBackprop>(
    Conv2DParams const& params) {
  const int n_tiles = params.window_rows * params.window_cols * params.channels;
  const int tile_size = params.out_rows * params.out_cols;
  return TileInfo{n_tiles, tile_size};
}
}  // namespace im2col
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_TILE_INFO_H_
