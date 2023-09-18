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
#ifndef PORTDNN_SRC_CONV2D_WINOGRAD_KERNELS_EXTRACT_INPUT_TRANSFORM_H_
#define PORTDNN_SRC_CONV2D_WINOGRAD_KERNELS_EXTRACT_INPUT_TRANSFORM_H_

#include "portdnn/accessor_types.h"

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/params.h"

#include "src/helpers/register_tile.h"
#include "src/helpers/tensor_index.h"
#include "src/helpers/vector_element.h"
#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"

#include "src/conv2d/winograd/kernels/tiles.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace winograd {

template <typename T, typename Index, int ChannelVector, int M, int N, int R,
          int S, typename ConvType, bool IsUSM>
struct ExtractInputTiles {
  using VecType = typename helpers::VectorType<T, ChannelVector>::type;

  ExtractInputTiles(Conv2DParams const& params, TileInfo const& tile_info,
                    ReadMem<T const, IsUSM> const& input,
                    WriteMem<T, IsUSM> const& output)
      : n_elems_{params.batch * tile_info.rows * tile_info.cols *
                 params.channels / ChannelVector},
        n_tiles_{tile_info.number * params.batch},
        n_tile_rows_{tile_info.rows},
        n_tile_cols_{tile_info.cols},
        n_in_cols_{params.in_cols},
        n_in_rows_{params.in_rows},
        n_channels_{params.channels},
        n_pad_cols_{params.pad_cols},
        n_pad_rows_{params.pad_rows},
        input_mem_{input},
        output_mem_{output} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<1> item) const {
    Index const index = item.get_id(0);
    if (index < n_elems_) {
      auto input_data = input_mem_.get_pointer();
      auto output_data = output_mem_.get_pointer();

      auto const tile_channel_idx =
          helpers::TensorIndexHelper<Index, false>::unflatten2d(
              index, n_channels_ / ChannelVector, n_channels_ / ChannelVector);
      Index const channel_idx = tile_channel_idx.s1 * ChannelVector;
      Index const tile_idx = tile_channel_idx.s0;

      auto const tile_tensor_idx =
          helpers::TensorIndexHelper<Index, false>::unflatten3d(
              tile_idx, n_tile_rows_, n_tile_rows_, n_tile_cols_, n_tile_cols_);
      Index const col_idx = tile_tensor_idx.s2;
      Index const row_idx = tile_tensor_idx.s1;
      Index const batch = tile_tensor_idx.s0;

      Index const cstart = col_idx * N - n_pad_cols_;
      Index const rstart = row_idx * M - n_pad_rows_;

      InputTile<VecType, M, N, R, S> inp(input_data, batch, rstart, n_in_rows_,
                                         cstart, n_in_cols_, channel_idx,
                                         n_channels_);

      OutputData<VecType, M, N, R, S>::write_transformed_input(
          output_data, tile_idx, channel_idx, n_tiles_, n_channels_,
          TransformedInputTile<VecType, M, N, R, S>{inp});
    }
  }

 private:
  Index const n_elems_;
  Index const n_tiles_;
  Index const n_tile_rows_;
  Index const n_tile_cols_;
  Index const n_in_cols_;
  Index const n_in_rows_;
  Index const n_channels_;
  Index const n_pad_cols_;
  Index const n_pad_rows_;
  ReadMem<T const, IsUSM> input_mem_;
  WriteMem<T, IsUSM> output_mem_;
};

template <typename T, typename Index, int ChannelVector, int M, int N, int R,
          int S, bool IsUSM>
struct ExtractInputTiles<T, Index, ChannelVector, M, N, R, S,
                         conv_type::FilterBackprop, IsUSM> {
  using VecType = typename helpers::VectorType<T, ChannelVector>::type;

  ExtractInputTiles(Conv2DParams const& params, TileInfo const& tile_info,
                    ReadMem<T const, IsUSM> const& input,
                    WriteMem<T, IsUSM> const& output)
      : n_elems_{params.batch * tile_info.rows * tile_info.cols *
                 params.channels / ChannelVector},
        n_tiles_{tile_info.number * params.batch},
        n_tile_rows_{tile_info.rows},
        n_tile_cols_{tile_info.cols},
        n_in_cols_{params.in_cols},
        n_in_rows_{params.in_rows},
        n_channels_{params.channels},
        n_pad_cols_{params.pad_cols},
        n_pad_rows_{params.pad_rows},
        input_mem_{input},
        output_mem_{output} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<1> item) const {
    Index const index = item.get_id(0);
    if (index < n_elems_) {
      auto input_data = input_mem_.get_pointer();
      auto output_data = output_mem_.get_pointer();

      auto const tile_channel_idx =
          helpers::TensorIndexHelper<Index, false>::unflatten2d(
              index, n_channels_ / ChannelVector, n_channels_ / ChannelVector);
      Index const channel_idx = tile_channel_idx.s1 * ChannelVector;
      Index const tile_idx = tile_channel_idx.s0;

      auto const tile_tensor_idx =
          helpers::TensorIndexHelper<Index, false>::unflatten3d(
              tile_idx, n_tile_rows_, n_tile_rows_, n_tile_cols_, n_tile_cols_);
      Index const col_idx = tile_tensor_idx.s2;
      Index const row_idx = tile_tensor_idx.s1;
      Index const batch = tile_tensor_idx.s0;

      Index const cstart = col_idx * S - n_pad_cols_;
      Index const rstart = row_idx * R - n_pad_rows_;

      InputTile<VecType, M, N, R, S> inp(input_data, batch, rstart, n_in_rows_,
                                         cstart, n_in_cols_, channel_idx,
                                         n_channels_);
      TransformedInputTile<VecType, M, N, R, S> trans{inp};

      OutputData<VecType, M, N, R, S>::write_transformed_input(
          output_data, tile_idx, channel_idx, n_tiles_, n_channels_, trans);
    }
  }

 private:
  Index const n_elems_;
  Index const n_tiles_;
  Index const n_tile_rows_;
  Index const n_tile_cols_;
  Index const n_in_cols_;
  Index const n_in_rows_;
  Index const n_channels_;
  Index const n_pad_cols_;
  Index const n_pad_rows_;
  ReadMem<T const, IsUSM> input_mem_;
  WriteMem<T, IsUSM> output_mem_;
};

}  // namespace winograd
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_SRC_CONV2D_WINOGRAD_KERNELS_EXTRACT_INPUT_TRANSFORM_H_
