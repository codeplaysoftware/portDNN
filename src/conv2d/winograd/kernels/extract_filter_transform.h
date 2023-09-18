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
#ifndef PORTDNN_SRC_CONV2D_WINOGRAD_KERNELS_EXTRACT_FILTER_TRANSFORM_H_
#define PORTDNN_SRC_CONV2D_WINOGRAD_KERNELS_EXTRACT_FILTER_TRANSFORM_H_

#include "portdnn/accessor_types.h"

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/params.h"
#include "portdnn/helpers/minmax.h"

#include "src/conv2d/winograd/kernels/tiles.h"
#include "src/helpers/tensor_index.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace winograd {

template <typename T, typename Index, int M, int N, int R, int S,
          typename ConvType, bool IsUSM>
struct ExtractFilterTiles {
  ExtractFilterTiles(Conv2DParams const& params, TileInfo const& /*unused*/,
                     ReadMem<T const, IsUSM> const& filter,
                     WriteMem<T, IsUSM> const& output)
      : n_tiles_{params.channels * params.features},
        n_channels_{params.channels},
        n_features_{params.features},
        filter_mem_{filter},
        output_mem_{output} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<1> item) const {
    Index const index = item.get_id(0);
    if (index < n_tiles_) {
      auto filter_data = filter_mem_.get_pointer();
      auto output_data = output_mem_.get_pointer();

      auto const channel_feature_idx =
          helpers::TensorIndexHelper<Index, false>::unflatten2d(
              index, n_features_, n_features_);
      Index const feature_idx = channel_feature_idx.s1;
      Index const channel_idx = channel_feature_idx.s0;

      FilterTile<T, M, N, R, S, ConvType> filter(
          filter_data, channel_idx, feature_idx, n_channels_, n_features_);
      TransformedFilterTile<T, M, N, R, S> transformed{filter};

      OutputData<T, M, N, R, S>::write_transformed_filter(
          output_data, feature_idx, channel_idx, n_features_, n_channels_,
          transformed);
    }
  }

 private:
  Index const n_tiles_;
  Index const n_channels_;
  Index const n_features_;
  ReadMem<T const, IsUSM> filter_mem_;
  WriteMem<T, IsUSM> output_mem_;
};

template <typename T, typename Index, int M, int N, int R, int S, bool IsUSM>
struct ExtractFilterTiles<T, Index, M, N, R, S, conv_type::InputBackprop,
                          IsUSM> {
  using ConvType = conv_type::InputBackprop;

  /*
   * Note that for the input backprop the features and channels in params have
   * been switched. params.channels_ are most packed in memory, which we expect
   * to be n_features_ in the filter. We switch these back in the constructor
   * so they are as expected.
   */
  ExtractFilterTiles(Conv2DParams const& params, TileInfo const& /*unused*/,
                     ReadMem<T const, IsUSM> const& filter,
                     WriteMem<T, IsUSM> const& output)
      : n_tiles_{params.channels * params.features},
        n_features_{params.channels},
        n_channels_{params.features},
        filter_mem_{filter},
        output_mem_{output} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<1> item) const {
    Index const index = item.get_id(0);
    if (index < n_tiles_) {
      auto filter_data = filter_mem_.get_pointer();
      auto output_data = output_mem_.get_pointer();

      auto const feature_channel_idx =
          helpers::TensorIndexHelper<Index, false>::unflatten2d(
              index, n_features_, n_features_);
      Index const feature_idx = feature_channel_idx.s1;
      Index const channel_idx = feature_channel_idx.s0;

      FilterTile<T, M, N, R, S, ConvType> filter(
          filter_data, channel_idx, feature_idx, n_channels_, n_features_);
      TransformedFilterTile<T, M, N, R, S> transformed{filter};

      OutputData<T, M, N, R, S>::write_transformed_filter(
          output_data, feature_idx, channel_idx, n_features_, n_channels_,
          transformed);
    }
  }

 private:
  Index const n_tiles_;
  Index const n_features_;
  Index const n_channels_;
  ReadMem<T const, IsUSM> filter_mem_;
  WriteMem<T, IsUSM> output_mem_;
};

template <typename T, typename Index, int M, int N, int R, int S, bool IsUSM>
struct ExtractFilterTiles<T, Index, M, N, R, S, conv_type::FilterBackprop,
                          IsUSM> {
  using ConvType = conv_type::FilterBackprop;

  ExtractFilterTiles(Conv2DParams const& params, TileInfo const& tile_info,
                     ReadMem<T const, IsUSM> const& filter,
                     WriteMem<T, IsUSM> const& output)
      : n_threads_{params.batch * tile_info.rows * tile_info.cols *
                   params.features},
        n_tiles_{tile_info.number * params.batch},
        n_tile_rows_{tile_info.rows},
        n_tile_cols_{tile_info.cols},
        n_window_rows_{params.window_rows},
        n_window_cols_{params.window_cols},
        n_features_{params.features},
        filter_mem_{std::move(filter)},
        output_mem_{std::move(output)} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<1> item) const {
    Index const index = item.get_id(0);
    if (index < n_threads_) {
      auto filter_data = filter_mem_.get_pointer();
      auto output_data = output_mem_.get_pointer();

      auto const tile_feature_idx =
          helpers::TensorIndexHelper<Index, false>::unflatten2d(
              index, n_features_, n_features_);
      Index const tile_idx = tile_feature_idx.s0;
      Index const feature = tile_feature_idx.s1;

      auto const tile_tensor_idx =
          helpers::TensorIndexHelper<Index, false>::unflatten3d(
              tile_idx, n_tile_rows_, n_tile_rows_, n_tile_cols_, n_tile_cols_);
      Index const col_idx = tile_tensor_idx.s2;
      Index const row_idx = tile_tensor_idx.s1;
      Index const batch = tile_tensor_idx.s0;

      Index const col = col_idx * S;
      Index const cend = helpers::min(col + N, n_window_cols_);

      Index const row = row_idx * R;
      Index const rend = helpers::min(row + M, n_window_rows_);

      Index const offset =
          ((batch * n_window_rows_ + row) * n_window_cols_ + col) *
              n_features_ +
          feature;
      SYCLOutputWindow<Index> w{rend - row, cend - col, offset};

      FilterTile<T, M, N, R, S, ConvType> filter(filter_data, w, n_window_cols_,
                                                 n_features_);
      TransformedFilterTile<T, M, N, R, S> transformed{filter};

      OutputData<T, M, N, R, S>::write_transformed_filter(
          output_data, feature, tile_idx, n_features_, n_tiles_, transformed);
    }
  }

 private:
  Index const n_threads_;
  Index const n_tiles_;
  Index const n_tile_rows_;
  Index const n_tile_cols_;
  Index const n_window_rows_;
  Index const n_window_cols_;
  Index const n_features_;
  ReadMem<T const, IsUSM> filter_mem_;
  WriteMem<T, IsUSM> output_mem_;
};

}  // namespace winograd
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_SRC_CONV2D_WINOGRAD_KERNELS_EXTRACT_FILTER_TRANSFORM_H_
