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
#ifndef PORTDNN_INCLUDE_INTERNAL_CONV2D_WINOGRAD_TILE_INFO_H_
#define PORTDNN_INCLUDE_INTERNAL_CONV2D_WINOGRAD_TILE_INFO_H_

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/params.h"
#include "portdnn/helpers/ratio.h"

#include <type_traits>

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace winograd {

/** Struct containing information about the tiles in a Winograd convolution. */
struct TileInfo {
  /** Number of tiles in row direction. */
  int rows;
  /** Number of tiles in col direction. */
  int cols;
  /** Total number of tiles. */
  int number;
};

/**
 * Compute the number of tiles required for the winograd transforms for given
 * tile sizes.
 *
 * \param params Kernel parameters for convolution
 * \return A TileInfo instance containing number of tiles required for the
 * convolution.
 */
template <typename ConvType, int OutTileRows, int OutTileCols, int R, int S,
          typename std::enable_if<
              !std::is_same<ConvType, conv_type::FilterBackprop>::value,
              int>::type = 0>
inline TileInfo get_tile_info(Conv2DParams const& params) {
  const int n_tile_rows =
      helpers::round_ratio_up_above_zero(params.out_rows, OutTileRows);
  const int n_tile_cols =
      helpers::round_ratio_up_above_zero(params.out_cols, OutTileCols);
  const int n_tiles = n_tile_rows * n_tile_cols;
  const TileInfo result{n_tile_rows, n_tile_cols, n_tiles};
  return result;
}

/** \copydoc get_tile_info() */
template <typename ConvType, int M, int N, int OutTileRows, int OutTileCols,
          typename std::enable_if<
              std::is_same<ConvType, conv_type::FilterBackprop>::value,
              int>::type = 0>
inline TileInfo get_tile_info(Conv2DParams const& params) {
  const int n_tile_rows =
      helpers::round_ratio_up_above_zero(params.window_rows, OutTileRows);
  const int n_tile_cols =
      helpers::round_ratio_up_above_zero(params.window_cols, OutTileCols);
  const int n_tiles = n_tile_rows * n_tile_cols;
  const TileInfo result{n_tile_rows, n_tile_cols, n_tiles};
  return result;
}

}  // namespace winograd
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_INTERNAL_CONV2D_WINOGRAD_TILE_INFO_H_
