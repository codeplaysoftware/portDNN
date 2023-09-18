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
#ifndef PORTDNN_SRC_CONV2D_TILED_TILE_INFO_H_
#define PORTDNN_SRC_CONV2D_TILED_TILE_INFO_H_

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/params.h"

#include "portdnn/helpers/ratio.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace tiled {

/**
 * Struct containing information about number of tiles required for a given
 * tiled convolution.
 */
struct TileInfo {
  /** Number of tiles in the row direction. */
  int n_rows;
  /** Number of tiles in the column direction. */
  int n_cols;
  /** Number of tiles in the channel direction. */
  int output_vectors;
};

/**
 * Get the number of tiles required for the convolution specified by the
 * parameters and tile sizes.
 *
 * \param params         Convolution parameters
 * \parma tile_rows      Number of rows covered in a single tile
 * \param tile_cols      Number of columns covered in a single tile
 * \param channel_vector Number of channels covered in a single tile
 * \param feature_vector Number of features covered in a single tile
 *
 * \return A TileInfo instance containing numbers of tile sizes for the given
 * convolution.
 */
template <typename ConvType>
inline TileInfo get_tile_info(Conv2DParams const& params, int tile_rows,
                              int tile_cols, int /*channel_vector*/,
                              int feature_vector) {
  auto rows = helpers::round_ratio_up_above_zero(params.out_rows, tile_rows);
  auto cols = helpers::round_ratio_up_above_zero(params.out_cols, tile_cols);
  auto output_vector = params.features / feature_vector;
  return {rows, cols, output_vector};
}

/** \copydoc get_tile_info() */
template <>
inline TileInfo get_tile_info<conv_type::InputBackprop>(
    Conv2DParams const& params, int tile_rows, int tile_cols,
    int channel_vector, int /*feature_vector*/) {
  auto rows = helpers::round_ratio_up_above_zero(params.in_rows, tile_rows);
  auto cols = helpers::round_ratio_up_above_zero(params.in_cols, tile_cols);
  auto output_vector = params.channels / channel_vector;
  return {rows, cols, output_vector};
}

}  // namespace tiled
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_SRC_CONV2D_TILED_TILE_INFO_H_
