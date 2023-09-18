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
#ifndef PORTDNN_SRC_CONV2D_TILED_OUTPUT_SIZE_H_
#define PORTDNN_SRC_CONV2D_TILED_OUTPUT_SIZE_H_

#include "portdnn/conv2d/params.h"
#include "portdnn/helpers/ratio.h"

namespace sycldnn {
namespace conv2d {
namespace internal {

template <typename ConvType, int TileRows, int Tile_cols,
          int ChannelVectorWidth, int FeatureVectorWidth>
struct TiledOutputSize {
  static size_t get(Conv2DParams const&) { return 0; }
};

template <int TileRows, int TileCols, int ChannelVectorWidth,
          int FeatureVectorWidth>
struct TiledOutputSize<conv_type::Forward, TileRows, TileCols,
                       ChannelVectorWidth, FeatureVectorWidth> {
  static size_t get(Conv2DParams const& params) {
    return params.batch *
           helpers::round_ratio_up_above_zero(params.out_rows, TileRows) *
           helpers::round_ratio_up_above_zero(params.out_cols, TileCols) *
           params.features / FeatureVectorWidth;
  }
};

template <int TileRows, int TileCols, int ChannelVectorWidth,
          int FeatureVectorWidth>
struct TiledOutputSize<conv_type::InputBackprop, TileRows, TileCols,
                       ChannelVectorWidth, FeatureVectorWidth> {
  static size_t get(Conv2DParams const& params) {
    return params.batch *
           helpers::round_ratio_up_above_zero(params.in_rows, TileRows) *
           helpers::round_ratio_up_above_zero(params.in_cols, TileCols) *
           params.channels / ChannelVectorWidth;
  }
};

}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_SRC_CONV2D_TILED_OUTPUT_SIZE_H_
