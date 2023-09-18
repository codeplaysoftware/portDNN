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
#ifndef PORTDNN_SRC_CONV2D_TILED_KERNELS_H_
#define PORTDNN_SRC_CONV2D_TILED_KERNELS_H_

#include "portdnn/accessor_types.h"

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/params.h"

#include "src/helpers/fast_div.h"
#include "src/helpers/math.h"
#include "src/helpers/register_tile.h"
#include "src/helpers/tensor_index.h"
#include "src/helpers/vector_element.h"
#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"
#include "src/helpers/window_index.h"

#include "src/conv2d/tiled/tile_info.h"
#include "src/conv2d/tiled/tiles.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace tiled {

template <typename T, typename Index, typename ConvType, int TileRows,
          int TileCols, int ChannelVectorWidth, int FeatureVectorWidth,
          bool UseFastDiv, int WindowRows, int WindowCols, int Stride,
          bool IsUSM>
struct TiledConv2D;

/**
 * Forward convolution using a tiled direct computation technique.
 *
 * This kernel can be vectorised in either the channels or the features. Both
 * significantly increases the number of registers required by the kernel, so is
 * unlikely to provide any additional performance. The feature vectorisation can
 * be controlled using the FeatureVectorWidth template. The channel
 * vectorisation needs the kernel to be modified so that the loop over the
 * channels is split into a vectorised part and a scalar part.
 */
template <typename T, typename Index, int OutTileRows, int OutTileCols,
          int ChannelVectorWidth, int FeatureVectorWidth, bool UseFastDiv,
          int WindowRows, int WindowCols, int Stride, bool IsUSM>
struct TiledConv2D<T, Index, conv_type::Forward, OutTileRows, OutTileCols,
                   ChannelVectorWidth, FeatureVectorWidth, UseFastDiv,
                   WindowRows, WindowCols, Stride, IsUSM> {
 private:
  using IndexDivType = typename fast_div::IndexDiv<Index, UseFastDiv>::type;
  static constexpr auto InputTileCols = (OutTileCols - 1) * Stride + WindowCols;
  static constexpr auto InputTileRows = (OutTileRows - 1) * Stride + WindowRows;
  using Input = InputRow<T, ChannelVectorWidth, InputTileCols>;
  using Filter = FilterTile<T, ChannelVectorWidth, FeatureVectorWidth,
                            WindowRows, WindowCols>;
  using Output = OutputTile<T, FeatureVectorWidth, OutTileRows, OutTileCols>;
  using InVecType = typename Input::VecType;
  using OutVecType = typename Output::VecType;

 public:
  TiledConv2D(ReadMem<T const, IsUSM> input, ReadMem<T const, IsUSM> filter,
              WriteMem<T, IsUSM> output, Conv2DParams const& params,
              TileInfo const& tile_info)
      : n_tile_cols_{tile_info.n_cols},
        n_tile_rows_{tile_info.n_rows},
        n_feature_vectors_{tile_info.output_vectors},
        div_feature_vectors_{n_feature_vectors_},
        div_n_tile_cols_{n_tile_cols_},
        div_n_tile_rows_{n_tile_rows_},
        n_elems_{params.batch * n_tile_rows_ * n_tile_cols_ *
                 n_feature_vectors_},
        channels_{params.channels},
        features_{params.features},
        batch_{params.batch},
        in_rows_{params.in_rows},
        in_cols_{params.in_cols},
        out_rows_{params.out_rows},
        out_cols_{params.out_cols},
        pad_rows_{params.pad_rows},
        pad_cols_{params.pad_cols},
        input_mem_{std::move(input)},
        filter_mem_{std::move(filter)},
        output_mem_{std::move(output)} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<1> item) const {
    Index const index = item.get_id(0);

    if (index < n_elems_) {
      auto input_data = input_mem_.get_pointer();
      auto filter_data = filter_mem_.get_pointer();
      auto output_data = output_mem_.get_pointer();

      auto const tensor_idx =
          helpers::TensorIndexHelper<Index, UseFastDiv>::unflatten4d(
              index, div_n_tile_rows_, n_tile_rows_, div_n_tile_cols_,
              n_tile_cols_, div_feature_vectors_, n_feature_vectors_);
      Index const feature = tensor_idx.s3 * FeatureVectorWidth;
      Index const col_idx = tensor_idx.s2 * OutTileCols;
      Index const row_idx = tensor_idx.s1 * OutTileRows;
      Index const batch = tensor_idx.s0;

      const auto col_window =
          helpers::in_window_from_output(col_idx, Stride, pad_cols_);
      const Index cstart = col_window.window_start;
      const auto row_window =
          helpers::in_window_from_output(row_idx, Stride, pad_rows_);
      const Index rstart = row_window.window_start;

      Output out_tile{};
      Index filter_offset = feature;
      Index input_channel_offset = batch * in_cols_ * in_rows_ * channels_;
      for (Index channel = 0; channel < channels_;
           channel += ChannelVectorWidth) {
        Filter filter_tile{filter_data, filter_offset, channels_, features_};

        Index input_offset =
            input_channel_offset + rstart * in_cols_ * channels_;
        for (Index i = 0; i < InputTileRows; ++i) {
          if (rstart + i >= 0 && rstart + i < in_rows_) {
            auto input_tile = Input::load_input_row(
                input_data, input_offset, cstart, in_cols_, channels_);
            convolve_tile(input_tile, filter_tile, out_tile, i);
          }
          input_offset += in_cols_ * channels_;
        }
        input_channel_offset += ChannelVectorWidth;
        filter_offset += ChannelVectorWidth * features_;
      }
      out_tile.write_out(output_data, batch, row_idx, out_rows_, col_idx,
                         out_cols_, feature, features_);
    }
  }

 private:
  void SNN_ALWAYS_INLINE convolve_tile(Input const& input, Filter const& filter,
                                       Output& output,
                                       int const row_idx) const {
    SNN_PRAGMA_UNROLL
    for (int out_row = 0; out_row < OutTileRows; ++out_row) {
      int const filter_row = row_idx - out_row * Stride;
      if (filter_row >= 0 && filter_row < WindowRows) {
        convolve_one_row(input, filter, output, out_row, filter_row);
      }
    }
  }

  void SNN_ALWAYS_INLINE convolve_one_row(Input const& input,
                                          Filter const& filter, Output& output,
                                          int const out_row,
                                          int const filter_row) const {
    int in_offset = 0;
    SNN_PRAGMA_UNROLL
    for (int out_col = 0; out_col < OutTileCols; ++out_col) {
      SNN_PRAGMA_UNROLL
      for (int filter_col = 0; filter_col < WindowCols; ++filter_col) {
        output.data(out_row, out_col) = forward_accumulate(
            input.data(in_offset + filter_col), filter, filter_row, filter_col,
            output.data(out_row, out_col));
      }
      in_offset += Stride;
    }
  }

  OutVecType SNN_ALWAYS_INLINE forward_accumulate(InVecType input,
                                                  Filter const& filter,
                                                  int const filter_row,
                                                  int const filter_col,
                                                  OutVecType value) const {
    SNN_PRAGMA_UNROLL
    for (int i = 0; i < ChannelVectorWidth; i++) {
      value =
          helpers::math::mad(OutVecType{helpers::vector_element::get(input, i)},
                             filter.data(filter_row, filter_col, i), value);
    }
    return value;
  }

  const Index n_tile_cols_;
  const Index n_tile_rows_;
  const Index n_feature_vectors_;
  const IndexDivType div_feature_vectors_;
  const IndexDivType div_n_tile_cols_;
  const IndexDivType div_n_tile_rows_;
  const Index n_elems_;
  const Index channels_;
  const Index features_;
  const Index batch_;
  const Index in_rows_;
  const Index in_cols_;
  const Index out_rows_;
  const Index out_cols_;
  const Index pad_rows_;
  const Index pad_cols_;
  const ReadMem<const T, IsUSM> input_mem_;
  const ReadMem<const T, IsUSM> filter_mem_;
  WriteMem<T, IsUSM> output_mem_;
};
template <typename T, typename Index, int OutTileRows, int OutTileCols,
          int ChannelVectorWidth, int FeatureVectorWidth, bool UseFastDiv,
          int WindowRows, int WindowCols, int Stride, bool IsUSM>
struct TiledConv2D<T, Index, conv_type::InputBackprop, OutTileRows, OutTileCols,
                   ChannelVectorWidth, FeatureVectorWidth, UseFastDiv,
                   WindowRows, WindowCols, Stride, IsUSM> {
 private:
  using IndexDivType = typename fast_div::IndexDiv<Index, UseFastDiv>::type;
  static constexpr auto InputTileCols = (OutTileCols + WindowCols - 1) / Stride;
  static constexpr auto InputTileRows = OutTileRows + WindowRows - 1;
  using Input = InputRow<T, FeatureVectorWidth, InputTileCols>;
  using Filter = FilterTile<T, ChannelVectorWidth, FeatureVectorWidth,
                            WindowRows, WindowCols>;
  using Output = OutputTile<T, ChannelVectorWidth, OutTileRows, OutTileCols>;
  using InVecType = typename Input::VecType;
  using OutVecType = typename Output::VecType;

 public:
  TiledConv2D(ReadMem<T const, IsUSM> input, ReadMem<T const, IsUSM> filter,
              WriteMem<T, IsUSM> output, Conv2DParams const& params,
              TileInfo const& tile_info)
      : n_tile_cols_{tile_info.n_cols},
        n_tile_rows_{tile_info.n_rows},
        n_channel_vectors_{tile_info.output_vectors},
        div_channels_{n_channel_vectors_},
        div_n_tile_cols_{n_tile_cols_},
        div_n_tile_rows_{n_tile_rows_},
        n_elems_{params.batch * n_tile_rows_ * n_tile_cols_ *
                 n_channel_vectors_},
        channels_{params.channels},
        features_{params.features},
        batch_{params.batch},
        in_rows_{params.in_rows},
        in_cols_{params.in_cols},
        out_rows_{params.out_rows},
        out_cols_{params.out_cols},
        pad_rows_{params.pad_rows},
        pad_cols_{params.pad_cols},
        input_mem_{std::move(input)},
        filter_mem_{std::move(filter)},
        output_mem_{std::move(output)} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<1> item) const {
    Index const index = item.get_id(0);

    if (index < n_elems_) {
      auto input_data = input_mem_.get_pointer();
      auto filter_data = filter_mem_.get_pointer();
      auto output_data = output_mem_.get_pointer();

      auto const tensor_idx =
          helpers::TensorIndexHelper<Index, UseFastDiv>::unflatten4d(
              index, div_n_tile_rows_, n_tile_rows_, div_n_tile_cols_,
              n_tile_cols_, div_channels_, n_channel_vectors_);
      Index const channel = tensor_idx.s3 * ChannelVectorWidth;
      Index const col_idx = tensor_idx.s2 * OutTileCols;
      Index const row_idx = tensor_idx.s1 * OutTileRows;
      Index const batch = tensor_idx.s0;

      const auto col_window =
          helpers::out_window_from_input(col_idx, Stride, pad_cols_);
      const Index cstart = col_window.window_start;
      const Index first_col = col_window.filter_start;
      const auto row_window =
          helpers::out_window_from_input(row_idx, Stride, pad_rows_);
      const Index rstart = row_window.window_start;
      const Index first_row = row_window.filter_start;

      Output out_tile{};

      Index filter_offset = channel * features_;
      Index input_feat_offset = batch * out_cols_ * out_rows_ * features_;
      for (Index feature = 0; feature < features_;
           feature += FeatureVectorWidth) {
        Filter filter_tile{filter_data, filter_offset, channels_, features_,
                           mirror_filter_tag{}};

        Index input_offset = input_feat_offset + rstart * out_cols_ * features_;
        for (Index r = rstart, i = first_row; i < InputTileRows;
             ++r, i += Stride) {
          if (r < out_rows_) {
            auto input_tile = Input::load_input_row(
                input_data, input_offset, cstart, out_cols_, features_);
            convolve_tile(input_tile, filter_tile, out_tile, i, first_col);
          }
          input_offset += out_cols_ * features_;
        }
        input_feat_offset += FeatureVectorWidth;
        filter_offset += FeatureVectorWidth;
      }
      out_tile.write_out(output_data, batch, row_idx, in_rows_, col_idx,
                         in_cols_, channel, channels_);
    }
  }

 private:
  void SNN_ALWAYS_INLINE convolve_tile(Input const& input, Filter const& filter,
                                       Output& output, int const row_idx,
                                       int const first_col) const {
    SNN_PRAGMA_UNROLL
    for (int out_row = 0; out_row < OutTileRows; ++out_row) {
      int const filter_row = row_idx - out_row;
      if (filter_row >= 0 && filter_row < WindowRows) {
        convolve_one_row(input, filter, output, out_row, filter_row, first_col);
      }
    }
  }
  void SNN_ALWAYS_INLINE convolve_one_row(Input const& input,
                                          Filter const& filter, Output& output,
                                          int const out_row,
                                          int const filter_row,
                                          int offset) const {
    SNN_PRAGMA_UNROLL
    for (int out_col = 0; out_col < OutTileCols; ++out_col) {
      auto padded_out = out_col - offset;
      auto in_offset = helpers::round_ratio_up_above_zero(padded_out, Stride);
      auto first_col = in_offset * Stride - padded_out;
      // first_col is always non-negative. If padded_out is negative, then
      // in_offset is zero so first_col = -padded_out > 0. If padded_out is
      // zero, then in_offset is zero and so is first_col. If padded_out is
      // positive, then in_offset*Stride is the multiple of Stride greater or
      // equal to padded_out.
      //
      // This allows us to start the following loop at zero, and use first_col
      // to shift each loop value. Then the shifted value will always be
      // greater or equal to zero.
      for (int filter_col = 0; filter_col < WindowCols; filter_col += Stride) {
        auto const shifted_filter_col = filter_col + first_col;
        if (shifted_filter_col < WindowCols) {
          output.data(out_row, out_col) = inputbackprop_accumulate(
              input.data(in_offset), filter, filter_row, shifted_filter_col,
              output.data(out_row, out_col));
          ++in_offset;
        }
      }
    }
  }
  OutVecType SNN_ALWAYS_INLINE inputbackprop_accumulate(
      InVecType input, Filter const& filter, int const filter_row,
      int const filter_col, OutVecType value) const {
    SNN_PRAGMA_UNROLL
    for (int i = 0; i < FeatureVectorWidth; ++i) {
      OutVecType filter_slice =
          slice_transpose(filter, filter_row, filter_col, i);
      value =
          helpers::math::mad(OutVecType{helpers::vector_element::get(input, i)},
                             filter_slice, value);
    }
    return value;
  }
  OutVecType SNN_ALWAYS_INLINE slice_transpose(Filter const& filter,
                                               int filter_row, int filter_col,
                                               int index) const {
    OutVecType output;
    SNN_PRAGMA_UNROLL
    for (int i = 0; i < ChannelVectorWidth; ++i) {
      helpers::vector_element::set(
          output, i,
          helpers::vector_element::get(filter.data(filter_row, filter_col, i),
                                       index));
    }
    return output;
  }

  const Index n_tile_cols_;
  const Index n_tile_rows_;
  const Index n_channel_vectors_;
  const IndexDivType div_channels_;
  const IndexDivType div_n_tile_cols_;
  const IndexDivType div_n_tile_rows_;
  const Index n_elems_;
  const Index channels_;
  const Index features_;
  const Index batch_;
  const Index in_rows_;
  const Index in_cols_;
  const Index out_rows_;
  const Index out_cols_;
  const Index pad_rows_;
  const Index pad_cols_;
  const ReadMem<const T, IsUSM> input_mem_;
  const ReadMem<const T, IsUSM> filter_mem_;
  WriteMem<T, IsUSM> output_mem_;
};

}  // namespace tiled
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_SRC_CONV2D_TILED_KERNELS_H_
