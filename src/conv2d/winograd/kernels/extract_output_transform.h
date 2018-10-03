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
#ifndef SYCLDNN_SRC_CONV2D_WINOGRAD_KERNELS_EXTRACT_OUTPUT_TRANSFORM_H_
#define SYCLDNN_SRC_CONV2D_WINOGRAD_KERNELS_EXTRACT_OUTPUT_TRANSFORM_H_

#include "sycldnn/accessor_types.h"

#include "sycldnn/conv2d/conv_type.h"
#include "sycldnn/conv2d/params.h"
#include "sycldnn/helpers/minmax.h"

#include "src/helpers/tensor_index.h"

#include "src/conv2d/winograd/kernels/tiles.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace winograd {

template <typename T, typename Index, int M, int N, int R, int S,
          typename ConvType, bool Accumulate = false>
struct ExtractOutputTiles {
  ExtractOutputTiles(Index in_offset, Index out_offset,
                     Conv2DParams const& params, TileInfo const& tile_info,
                     ReadAccessor<T const> const& input,
                     WriteAccessor<T> const& output)
      : in_offset_{in_offset},
        out_offset_{out_offset},
        n_threads_{params.batch * tile_info.rows * tile_info.cols *
                   params.features},
        n_tiles_{tile_info.number * params.batch},
        n_tile_rows_{tile_info.rows},
        n_tile_cols_{tile_info.cols},
        n_out_rows_{params.out_rows},
        n_out_cols_{params.out_cols},
        n_features_{params.features},
        input_accessor_{input},
        output_accessor_{output} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<1> item) {
    Index const index = item.get_id(0);
    if (index < n_threads_) {
      T const* input_data = input_accessor_.get_pointer().get() + in_offset_;
      T* output_data = output_accessor_.get_pointer().get() + out_offset_;

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

      IntermediateTile<T, M, N, R, S> tmp{input_data, tile_idx, n_tiles_,
                                          feature, n_features_};

      Index const col = col_idx * N;
      Index const cend = helpers::min(col + N, n_out_cols_);

      Index const row = row_idx * M;
      Index const rend = helpers::min(row + M, n_out_rows_);

      Index const offset =
          ((batch * n_out_rows_ + row) * n_out_cols_ + col) * n_features_ +
          feature;

      SYCLOutputWindow<Index> out_w{rend - row, cend - col, offset};

      OutputData<T, M, N, R, S>::write_output(output_data, out_w, n_out_cols_,
                                              n_features_,
                                              OutputTile<T, M, N, R, S>{tmp});
    }
  }

 private:
  Index const in_offset_;
  Index const out_offset_;
  Index const n_threads_;
  Index const n_tiles_;
  Index const n_tile_rows_;
  Index const n_tile_cols_;
  Index const n_out_rows_;
  Index const n_out_cols_;
  Index const n_features_;
  ReadAccessor<T const> input_accessor_;
  WriteAccessor<T> output_accessor_;
};

template <typename T, typename Index, int M, int N, int R, int S,
          bool Accumulate>
struct ExtractOutputTiles<T, Index, M, N, R, S, conv_type::FilterBackprop,
                          Accumulate> {
  ExtractOutputTiles(Index in_offset, Index out_offset,
                     Conv2DParams const& params, TileInfo const& /*unused*/,
                     ReadAccessor<T const> const& input,
                     WriteAccessor<T> const& output)
      : in_offset_{in_offset},
        out_offset_{out_offset},
        n_threads_{params.features * params.channels},
        n_features_{params.features},
        n_channels_{params.channels},
        input_accessor_{input},
        output_accessor_{output} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<1> item) {
    Index const index = item.get_id(0);
    if (index < n_threads_) {
      T const* input_data = input_accessor_.get_pointer().get() + in_offset_;
      T* output_data = output_accessor_.get_pointer().get() + out_offset_;

      auto const channel_feature_idx =
          helpers::TensorIndexHelper<Index, false>::unflatten2d(
              index, n_features_, n_features_);
      Index const channel = channel_feature_idx.s0;
      Index const feature = channel_feature_idx.s1;

      IntermediateTile<T, M, N, R, S> tmp{input_data, channel, n_channels_,
                                          feature, n_features_};
      OutputData<T, M, N, R, S>::template write_filter_output<Accumulate>(
          output_data, channel, feature, n_channels_, n_features_,
          OutputTile<T, M, N, R, S>{tmp});
    }
  }

 private:
  Index const in_offset_;
  Index const out_offset_;
  Index const n_threads_;
  Index const n_features_;
  Index const n_channels_;
  ReadAccessor<T const> input_accessor_;
  WriteAccessor<T> output_accessor_;
};

}  // namespace winograd
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_CONV2D_WINOGRAD_KERNELS_EXTRACT_OUTPUT_TRANSFORM_H_