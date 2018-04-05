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
#ifndef SYCLDNN_SRC_CONV2D_TILED_KERNELS_H_
#define SYCLDNN_SRC_CONV2D_TILED_KERNELS_H_

#include "sycldnn/accessor_types.h"

#include "sycldnn/conv2d/conv_type.h"
#include "sycldnn/conv2d/params.h"

#include "src/helpers/fast_div.h"
#include "src/helpers/math.h"
#include "src/helpers/register_tile.h"
#include "src/helpers/tensor_index.h"
#include "src/helpers/vector_element.h"
#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"
#include "src/helpers/window_index.h"

namespace sycldnn {
namespace conv2d {
namespace tiled {
struct check_bounds_tag {};
struct mirror_filter_tag {};
/** A 1 x Width row from the input tensor. */
template <typename T, int ChannelVector, int Width>
struct InputRow final
    : public helpers::RegisterTile1D<
          typename helpers::VectorType<T, ChannelVector>::type, Width> {
  using VecType = typename helpers::VectorType<T, ChannelVector>::type;
  using helpers::RegisterTile1D<VecType, Width>::data;
  InputRow() = default;
  template <typename _T, typename Index>
  inline SNN_ALWAYS_INLINE InputRow(_T const* input, Index const row,
                                    Index const /*n_rows*/, Index const col,
                                    Index const n_cols, Index const channel,
                                    Index const n_channels) {
    Index idx = (row * n_cols + col) * n_channels + channel;
    for (int i = 0; i < Width; ++i) {
      data(i) = helpers::io::Load<VecType>()(input, idx);
      idx += n_channels;
    }
  }
  template <typename _T, typename Index>
  inline SNN_ALWAYS_INLINE InputRow(_T const* input, Index const row,
                                    Index const /*n_rows*/, Index const col,
                                    Index const n_cols, Index const channel,
                                    Index const n_channels, check_bounds_tag) {
    Index idx = (row * n_cols + col) * n_channels + channel;
    for (int i = 0; i < Width; ++i) {
      data(i) = (col + i < 0 || col + i >= n_cols)
                    ? VecType{static_cast<T>(0)}
                    : helpers::io::Load<VecType>()(input, idx);
      idx += n_channels;
    }
  }
};
/** A WindowRows x WindowCols tile from the filter tensor. */
template <typename T, int ChannelVector, int FeatureVector, int WindowRows,
          int WindowCols>
struct FilterTile : public helpers::RegisterTile3D<
                        typename helpers::VectorType<T, FeatureVector>::type,
                        WindowRows, WindowCols, ChannelVector> {
  using VecType = typename helpers::VectorType<T, FeatureVector>::type;
  using helpers::RegisterTile3D<VecType, WindowRows, WindowCols,
                                ChannelVector>::data;
  template <typename _T, typename Index>
  inline SNN_ALWAYS_INLINE FilterTile(_T const* const input,
                                      Index const channel,
                                      Index const n_channels,
                                      Index const feature,
                                      Index const n_features) {
    Index row_idx = 0;
    for (int i = 0; i < WindowRows; ++i) {
      Index col_idx = row_idx;
      for (int j = 0; j < WindowCols; ++j) {
        Index ch_idx = col_idx + channel * n_features + feature;
        for (int ch_v = 0; ch_v < ChannelVector; ++ch_v) {
          data(i, j, ch_v) = helpers::io::Load<VecType>()(input, ch_idx);
          ch_idx += n_features;
        }
        col_idx += n_channels * n_features;
      }
      row_idx += WindowCols * n_channels * n_features;
    }
  }
  template <typename _T, typename Index>
  inline SNN_ALWAYS_INLINE FilterTile(
      _T const* const input, Index const channel, Index const n_channels,
      Index const feature, Index const n_features, mirror_filter_tag) {
    Index row_idx = 0;
    for (int i = 0; i < WindowRows; ++i) {
      Index col_idx = row_idx;
      for (int j = 0; j < WindowCols; ++j) {
        Index ch_idx = col_idx + channel * n_features + feature;
        for (int ch_v = 0; ch_v < ChannelVector; ++ch_v) {
          data(WindowRows - 1 - i, WindowCols - 1 - j, ch_v) =
              helpers::io::Load<VecType>()(input, ch_idx);
          ch_idx += n_features;
        }
        col_idx += n_channels * n_features;
      }
      row_idx += WindowCols * n_channels * n_features;
    }
  }
};
/* An OutTileRows x OutTileCols tile to collect output results. */
template <typename T, int VectorWidth, int OutTileRows, int OutTileCols>
struct OutputTile final
    : helpers::RegisterTile2D<
          typename helpers::VectorType<T, VectorWidth>::type, OutTileRows,
          OutTileCols> {
  using VecType = typename helpers::VectorType<T, VectorWidth>::type;
  using helpers::RegisterTile2D<VecType, OutTileRows, OutTileCols>::data;
  template <typename _T, typename Index>
  inline SNN_ALWAYS_INLINE void write_out(
      _T* output, Index const batch, Index const out_row, Index const n_rows,
      Index const out_col, Index const n_cols, Index const feature,
      Index const n_features, check_bounds_tag) {
    static_assert(!std::is_const<_T>::value,
                  "Cannot write values to a pointer to const types.");
    Index const offset =
        ((batch * n_rows + out_row) * n_cols + out_col) * n_features + feature;
    output += offset;
    Index row_idx = 0;
    for (Index tile_row = 0; tile_row < OutTileRows; ++tile_row) {
      if (tile_row < n_rows - out_row) {
        Index idx = row_idx;
        for (Index tile_col = 0; tile_col < OutTileCols; ++tile_col) {
          if (tile_col < n_cols - out_col) {
            helpers::io::Store<VecType>()(output, idx,
                                          data(tile_row, tile_col));
            idx += n_features;
          }
        }
        row_idx += n_cols * n_features;
      }
    }
  }
  template <typename _T, typename Index>
  inline SNN_ALWAYS_INLINE void write_out(
      _T* output, Index const batch, Index const out_row, Index const n_rows,
      Index const out_col, Index const n_cols, Index const feature,
      Index const n_features) {
    static_assert(!std::is_const<_T>::value,
                  "Cannot write values to a pointer to const types.");
    Index const offset =
        ((batch * n_rows + out_row) * n_cols + out_col) * n_features + feature;
    output += offset;
    Index row_idx = 0;
    for (Index tile_row = 0; tile_row < OutTileRows; ++tile_row) {
      Index idx = row_idx;
      for (Index tile_col = 0; tile_col < OutTileCols; ++tile_col) {
        helpers::io::Store<VecType>()(output, idx, data(tile_row, tile_col));
        idx += n_features;
      }
      row_idx += n_cols * n_features;
    }
  }
};
template <typename T, typename Index, typename ConvType, int TileRows,
          int TileCols, int ChannelVectorWidth, int FeatureVectorWidth,
          bool UseFastDiv, int WindowRows, int WindowCols, int Stride = 0>
struct TiledConv2D {
  TiledConv2D(Conv2DParams const& /*params*/,
              ReadAccessor<T const> const /*input*/,
              ReadAccessor<T const> const /*kernel*/,
              WriteAccessor<T> /*output*/) {}
  inline SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> /*item*/) {}
};
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
          int WindowRows, int WindowCols, int Stride>
struct TiledConv2D<T, Index, conv_type::Forward, OutTileRows, OutTileCols,
                   ChannelVectorWidth, FeatureVectorWidth, UseFastDiv,
                   WindowRows, WindowCols, Stride> {
  using IndexDivType = typename fast_div::IndexDiv<Index, UseFastDiv>::type;
  static constexpr auto InputTileCols = (OutTileCols - 1) * Stride + WindowCols;
  static constexpr auto InputTileRows = (OutTileRows - 1) * Stride + WindowRows;
  using Input = InputRow<T, ChannelVectorWidth, InputTileCols>;
  using Filter = FilterTile<T, ChannelVectorWidth, FeatureVectorWidth,
                            WindowRows, WindowCols>;
  using Output = OutputTile<T, FeatureVectorWidth, OutTileRows, OutTileCols>;
  using InVecType = typename Input::VecType;
  using OutVecType = typename Output::VecType;

  TiledConv2D(Conv2DParams const& params, ReadAccessor<T const> input,
              ReadAccessor<T const> kernel, WriteAccessor<T> output)
      : n_tile_cols_{helpers::round_ratio_up_above_zero(params.out_cols,
                                                        OutTileCols)},
        n_tile_rows_{
            helpers::round_ratio_up_above_zero(params.out_rows, OutTileRows)},
        n_elems_{params.batch * n_tile_rows_ * n_tile_cols_ * params.features /
                 FeatureVectorWidth},
        n_feature_vectors_{params.features / FeatureVectorWidth},
        div_feature_vectors_{n_feature_vectors_},
        div_n_tile_cols_{n_tile_cols_},
        div_n_tile_rows_{n_tile_rows_},
        channels_{params.channels},
        features_{params.features},
        batch_{params.batch},
        in_rows_{params.in_rows},
        in_cols_{params.in_cols},
        out_rows_{params.out_rows},
        out_cols_{params.out_cols},
        pad_rows_{params.pad_rows},
        pad_cols_{params.pad_cols},
        input_accessor_{std::move(input)},
        filter_accessor_{std::move(kernel)},
        output_accessor_{std::move(output)} {}

  inline SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get_id(0);
    const Index range = item.get_range().get(0);

    for (; index < n_elems_; index += range) {
      const T* input_data = input_accessor_.get_pointer().get();
      const T* filter_data = filter_accessor_.get_pointer().get();
      T* output_data = output_accessor_.get_pointer().get();

      const auto tensor_idx =
          helpers::TensorIndexHelper<Index, UseFastDiv>::unflatten4d(
              index, div_n_tile_rows_, n_tile_rows_, div_n_tile_cols_,
              n_tile_cols_, div_feature_vectors_, n_feature_vectors_);
      const Index feature = tensor_idx.s3 * FeatureVectorWidth;
      const Index col_idx = tensor_idx.s2 * OutTileCols;
      const Index row_idx = tensor_idx.s1 * OutTileRows;
      const Index batch = tensor_idx.s0;

      const auto col_window =
          helpers::in_window_from_output(col_idx, Stride, pad_cols_);
      const Index cstart = col_window.window_start;
      const auto row_window =
          helpers::in_window_from_output(row_idx, Stride, pad_rows_);
      const Index rstart = row_window.window_start;

      Output out_tile{};
      const T* input_data_n =
          input_data + batch * in_cols_ * in_rows_ * channels_;
      for (Index channel = 0; channel < channels_;
           channel += ChannelVectorWidth) {
        Filter filter_tile{filter_data, channel, channels_, feature, features_};
        for (Index r = rstart, i = 0; i < InputTileRows; ++r, ++i) {
          if (r >= 0 && r < in_rows_) {
            Input input_tile;
            if (cstart >= 0 && cstart + InputTileCols < in_cols_) {
              input_tile = Input{input_data_n, r,       in_rows_, cstart,
                                 in_cols_,     channel, channels_};
            } else {
              input_tile =
                  Input{input_data_n, r,       in_rows_,  cstart,
                        in_cols_,     channel, channels_, check_bounds_tag{}};
            }
            convolve_tile(input_tile, filter_tile, out_tile, i);
          }
        }
      }
      if (row_idx + OutTileRows < out_rows_ &&
          col_idx + OutTileCols < out_cols_) {
        out_tile.write_out(output_data, batch, row_idx, out_rows_, col_idx,
                           out_cols_, feature, features_);
      } else {
        out_tile.write_out(output_data, batch, row_idx, out_rows_, col_idx,
                           out_cols_, feature, features_, check_bounds_tag{});
      }
    }
  }

 private:
  inline SNN_ALWAYS_INLINE void convolve_tile(Input const& input,
                                              Filter const& filter,
                                              Output& output,
                                              int const row_idx) {
    SNN_PRAGMA_UNROLL
    for (int out_row = 0; out_row < OutTileRows; ++out_row) {
      int const filter_row = row_idx - out_row * Stride;
      if (filter_row >= 0 && filter_row < WindowRows) {
        convolve_one_row(input, filter, output, out_row, filter_row);
      }
    }
  }
  inline SNN_ALWAYS_INLINE void convolve_one_row(Input const& input,
                                                 Filter const& filter,
                                                 Output& output,
                                                 int const out_row,
                                                 int const filter_row) {
    int in_offset = 0;
    for (int out_col = 0; out_col < OutTileCols; ++out_col) {
      for (int filter_col = 0; filter_col < WindowCols; ++filter_col) {
        output.data(out_row, out_col) = forward_accumulate(
            input.data(in_offset + filter_col), filter, filter_row, filter_col,
            output.data(out_row, out_col));
      }
      in_offset += Stride;
    }
  }
  inline SNN_ALWAYS_INLINE OutVecType forward_accumulate(InVecType input,
                                                         Filter const& filter,
                                                         int const filter_row,
                                                         int const filter_col,
                                                         OutVecType value) {
    for (int i = 0; i < ChannelVectorWidth; i++) {
      value =
          helpers::math::mad(OutVecType{helpers::vector_element::get(input, i)},
                             filter.data(filter_row, filter_col, i), value);
    }
    return value;
  }

  const Index n_tile_cols_;
  const Index n_tile_rows_;
  const Index n_elems_;
  const Index n_feature_vectors_;
  const IndexDivType div_feature_vectors_;
  const IndexDivType div_n_tile_cols_;
  const IndexDivType div_n_tile_rows_;
  const Index channels_;
  const Index features_;
  const Index batch_;
  const Index in_rows_;
  const Index in_cols_;
  const Index out_rows_;
  const Index out_cols_;
  const Index pad_rows_;
  const Index pad_cols_;
  const ReadAccessor<const T> input_accessor_;
  const ReadAccessor<const T> filter_accessor_;
  WriteAccessor<T> output_accessor_;
};
template <typename T, typename Index, int OutTileRows, int OutTileCols,
          int ChannelVectorWidth, int FeatureVectorWidth, bool UseFastDiv,
          int WindowRows, int WindowCols, int Stride>
struct TiledConv2D<T, Index, conv_type::InputBackprop, OutTileRows, OutTileCols,
                   ChannelVectorWidth, FeatureVectorWidth, UseFastDiv,
                   WindowRows, WindowCols, Stride> {
  using IndexDivType = typename fast_div::IndexDiv<Index, UseFastDiv>::type;
  static constexpr auto InputTileCols = (OutTileCols + WindowCols - 1) / Stride;
  static constexpr auto InputTileRows = OutTileRows + WindowRows - 1;
  using Input = InputRow<T, FeatureVectorWidth, InputTileCols>;
  using Filter = FilterTile<T, ChannelVectorWidth, FeatureVectorWidth,
                            WindowRows, WindowCols>;
  using Output = OutputTile<T, ChannelVectorWidth, OutTileRows, OutTileCols>;
  using InVecType = typename Input::VecType;
  using OutVecType = typename Output::VecType;

  TiledConv2D(Conv2DParams const& params, ReadAccessor<T const> input,
              ReadAccessor<T const> kernel, WriteAccessor<T> output)
      : n_tile_cols_{helpers::round_ratio_up_above_zero(params.in_cols,
                                                        OutTileCols)},
        n_tile_rows_{
            helpers::round_ratio_up_above_zero(params.in_rows, OutTileRows)},
        n_elems_{params.batch * n_tile_rows_ * n_tile_cols_ * params.channels /
                 ChannelVectorWidth},
        div_channels_{params.channels / ChannelVectorWidth},
        div_n_tile_cols_{n_tile_cols_},
        div_n_tile_rows_{n_tile_rows_},
        channels_{params.channels},
        features_{params.features},
        batch_{params.batch},
        in_rows_{params.in_rows},
        in_cols_{params.in_cols},
        out_rows_{params.out_rows},
        out_cols_{params.out_cols},
        pad_rows_{params.pad_rows},
        pad_cols_{params.pad_cols},
        input_accessor_{std::move(input)},
        filter_accessor_{std::move(kernel)},
        output_accessor_{std::move(output)} {}

  inline SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get_id(0);
    const Index range = item.get_range().get(0);

    for (; index < n_elems_; index += range) {
      const T* input_data = input_accessor_.get_pointer().get();
      const T* filter_data = filter_accessor_.get_pointer().get();
      T* output_data = output_accessor_.get_pointer().get();

      const auto tensor_idx =
          helpers::TensorIndexHelper<Index, UseFastDiv>::unflatten4d(
              index, div_n_tile_rows_, n_tile_rows_, div_n_tile_cols_,
              n_tile_cols_, div_channels_, channels_ / ChannelVectorWidth);
      const Index channel = tensor_idx.s3 * ChannelVectorWidth;
      const Index col_idx = tensor_idx.s2 * OutTileCols;
      const Index row_idx = tensor_idx.s1 * OutTileRows;
      const Index batch = tensor_idx.s0;

      const auto col_window =
          helpers::out_window_from_input(col_idx, Stride, pad_cols_);
      const Index cstart = col_window.window_start;
      const Index first_col = col_window.filter_start;
      const auto row_window =
          helpers::out_window_from_input(row_idx, Stride, pad_rows_);
      const Index rstart = row_window.window_start;
      const Index first_row = row_window.filter_start;

      Output out_tile{};
      const T* input_data_n =
          input_data + batch * out_cols_ * out_rows_ * features_;
      for (Index feature = 0; feature < features_;
           feature += FeatureVectorWidth) {
        Filter filter_tile{filter_data, channel,   channels_,
                           feature,     features_, mirror_filter_tag{}};

        for (Index r = rstart, i = first_row; i < InputTileRows;
             ++r, i += Stride) {
          if (r < out_rows_) {
            Input input_tile;
            if (cstart + InputTileCols < out_cols_) {
              input_tile = Input{input_data_n, r,       out_rows_, cstart,
                                 out_cols_,    feature, features_};
            } else {
              input_tile =
                  Input{input_data_n, r,       out_rows_, cstart,
                        out_cols_,    feature, features_, check_bounds_tag{}};
            }
            convolve_tile(input_tile, filter_tile, out_tile, i, first_col);
          }
        }
      }
      if (row_idx + OutTileRows < in_rows_ &&
          col_idx + OutTileCols < in_cols_) {
        out_tile.write_out(output_data, batch, row_idx, in_rows_, col_idx,
                           in_cols_, channel, channels_);
      } else {
        out_tile.write_out(output_data, batch, row_idx, in_rows_, col_idx,
                           in_cols_, channel, channels_, check_bounds_tag{});
      }
    }
  }

 private:
  inline SNN_ALWAYS_INLINE void convolve_tile(Input const& input,
                                              Filter const& filter,
                                              Output& output, int const row_idx,
                                              int const first_col) {
    SNN_PRAGMA_UNROLL
    for (int out_row = 0; out_row < OutTileRows; ++out_row) {
      int const filter_row = row_idx - out_row;
      if (filter_row >= 0 && filter_row < WindowRows) {
        convolve_one_row(input, filter, output, out_row, filter_row, first_col);
      }
    }
  }
  inline SNN_ALWAYS_INLINE void convolve_one_row(
      Input const& input, Filter const& filter, Output& output,
      int const out_row, int const filter_row, int offset) {
    int first_col = 0;
    for (int out_col = 0; out_col < OutTileCols; ++out_col) {
      int in_offset = helpers::round_ratio_up(out_col - offset, Stride);
      for (int filter_col = first_col; filter_col < WindowCols;
           filter_col += Stride, ++in_offset) {
        if (in_offset >= 0) {
          output.data(out_row, out_col) = inputbackprop_accumulate(
              input.data(in_offset), filter, filter_row, filter_col,
              output.data(out_row, out_col));
        }
      }
      first_col--;
      if (first_col < 0) {
        first_col = Stride - 1;
      }
    }
  }
  inline SNN_ALWAYS_INLINE OutVecType inputbackprop_accumulate(
      InVecType input, Filter const& filter, int const filter_row,
      int const filter_col, OutVecType value) {
    for (int i = 0; i < FeatureVectorWidth; ++i) {
      OutVecType filter_slice =
          slice_transpose(filter, filter_row, filter_col, i);
      value =
          helpers::math::mad(OutVecType{helpers::vector_element::get(input, i)},
                             filter_slice, value);
    }
    return value;
  }
  inline SNN_ALWAYS_INLINE OutVecType slice_transpose(Filter const& filter,
                                                      int filter_row,
                                                      int filter_col,
                                                      int index) {
    OutVecType output;
    for (int i = 0; i < ChannelVectorWidth; ++i) {
      helpers::vector_element::set(
          output, i, helpers::vector_element::get(
                         filter.data(filter_row, filter_col, i), index));
    }
    return output;
  }

  const Index n_tile_cols_;
  const Index n_tile_rows_;
  const Index n_elems_;
  const IndexDivType div_channels_;
  const IndexDivType div_n_tile_cols_;
  const IndexDivType div_n_tile_rows_;
  const Index channels_;
  const Index features_;
  const Index batch_;
  const Index in_rows_;
  const Index in_cols_;
  const Index out_rows_;
  const Index out_cols_;
  const Index pad_rows_;
  const Index pad_cols_;
  const ReadAccessor<const T> input_accessor_;
  const ReadAccessor<const T> filter_accessor_;
  WriteAccessor<T> output_accessor_;
};
}  // namespace tiled
}  // namespace conv2d
}  // namespace sycldnn
#endif  // SYCLDNN_SRC_CONV2D_TILEL_KERNELS_H_
