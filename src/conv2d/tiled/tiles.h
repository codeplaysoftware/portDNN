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
#ifndef SYCLDNN_SRC_CONV2D_TILED_TILES_H_
#define SYCLDNN_SRC_CONV2D_TILED_TILES_H_

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
#include "sycldnn/format_type.h"

#include "src/conv2d/tiled/tile_info.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace tiled {

struct check_bounds_tag {};

/** A 1 x Width row from the input tensor. */
template <typename T, int VectorWidth, int Width, DataFormat Layout>
struct InputRow;

template <typename T, int ChannelVector, int Width>
struct InputRow<T, ChannelVector, Width, DataFormat::NHWC> final
    : public helpers::RegisterTile1D<
          typename helpers::VectorType<T, ChannelVector>::type, Width> {
 public:
  using VecType = typename helpers::VectorType<T, ChannelVector>::type;
  using helpers::RegisterTile1D<VecType, Width>::data;

  /**
   * Input row factory method. Will load the input data specified by row, col
   * and channel into an InputRow tile from the given multi pointer.
   */
  template <typename Index, MULTI_PTR_TEMPLATE_DECL>
  static InputRow SNN_ALWAYS_INLINE
  load_input_row(cl::sycl::multi_ptr<T const, MULTI_PTR_TEMPLATE> input,
                 Index const offset, Index const col, Index const n_cols,
                 Index const n_channels) {
    if (col >= 0 && col + Width < n_cols) {
      return {input, offset, col, n_cols, n_channels};
    } else {
      return {input, offset, col, n_cols, n_channels, check_bounds_tag{}};
    };
  }

 private:
  template <typename Index, MULTI_PTR_TEMPLATE_DECL>
  SNN_ALWAYS_INLINE InputRow(
      cl::sycl::multi_ptr<T const, MULTI_PTR_TEMPLATE> input,
      Index const offset, Index const col, Index const /*n_cols*/,
      Index const n_channels) {
    Index idx = offset + col * n_channels;
    SNN_PRAGMA_UNROLL
    for (int i = 0; i < Width; ++i) {
      data(i) = helpers::io::Load<VecType>()(input, idx);
      idx += n_channels;
    }
  }

  template <typename Index, MULTI_PTR_TEMPLATE_DECL>
  SNN_ALWAYS_INLINE InputRow(
      cl::sycl::multi_ptr<T const, MULTI_PTR_TEMPLATE> input,
      Index const offset, Index const col, Index const n_cols,
      Index const n_channels, check_bounds_tag) {
    Index idx = offset + col * n_channels;
    SNN_PRAGMA_UNROLL
    for (int i = 0; i < Width; ++i) {
      data(i) = (col + i < 0 || col + i >= n_cols)
                    ? VecType{0}
                    : helpers::io::Load<VecType>()(input, idx);
      idx += n_channels;
    }
  }
};

template <typename T, int Dummy, int Width>
struct InputRow<T, Dummy, Width, DataFormat::NCHW> final
    : public helpers::RegisterTile1D<T, Width> {
 private:
  static constexpr int VectorWidth = helpers::io::get_vec_size(Width);
  static_assert(Width % VectorWidth == 0 && "Bad tile width/vector width combination.");
  using VecType = typename helpers::VectorType<T, VectorWidth>::type;
  static constexpr auto NumVecElems = Width / VectorWidth;
 public:
  using helpers::RegisterTile1D<T, Width>::data;

  /**
   * Input row factory method. Will load the input data specified by row, col
   * and channel into an InputRow tile from the given multi pointer.
   */
  template <typename Index, MULTI_PTR_TEMPLATE_DECL>
  static InputRow SNN_ALWAYS_INLINE
  load_input_row(cl::sycl::multi_ptr<T const, MULTI_PTR_TEMPLATE> input,
                 Index const offset, Index const col, Index const n_cols,
                 Index const n_channels) {
    if (col >= 0 && col + Width < n_cols) {
      return {input, offset, col, n_cols, n_channels};
    } else {
      return {input, offset, col, n_cols, n_channels, check_bounds_tag{}};
    };
  }

 private:
  template <typename Index, MULTI_PTR_TEMPLATE_DECL>
  SNN_ALWAYS_INLINE InputRow(
      cl::sycl::multi_ptr<T const, MULTI_PTR_TEMPLATE> input,
      Index const offset, Index const col, Index const /*n_cols*/,
      Index const /*n_channels*/) {
    Index idx = offset + col;
    SNN_PRAGMA_UNROLL
    for (int i = 0; i < Width; i+=VectorWidth) {
      VecType input_vals = helpers::io::Load<VecType>()(input, idx);
      for (int v = 0; v < VectorWidth; ++v){
        data(i + v) = helpers::vector_element::get(input_vals, v);
      }
      idx += VectorWidth;
    }
  }

  template <typename Index, MULTI_PTR_TEMPLATE_DECL>
  SNN_ALWAYS_INLINE InputRow(
      cl::sycl::multi_ptr<T const, MULTI_PTR_TEMPLATE> input,
      Index const offset, Index const col, Index const n_cols,
      Index const n_channels, check_bounds_tag) {
    SNN_UNUSED_VAR(n_channels);
    Index idx = offset + col;
    SNN_PRAGMA_UNROLL
    for (int i = 0; i < Width; ++i) {
      data(i) = (col + i < 0 || col + i >= n_cols)
                    ? static_cast<sycl::vec<T,1>>(0)
                    : helpers::io::Load<sycl::vec<T,1>>()(input, idx);
      ++idx;
    }
  }
};

template <typename T, int ChannelVector, int FeatureVector, int WindowRows,
          int WindowCols, FilterFormat Layout, typename ConvType>
struct FilterTile;

/** HWCF: A 3D tile (WindowRows x WindowCols x ChannelVector) from the filter tensor,
 * composed of sycl::vec<T,FeatureVector> elements.
*/
template <typename T, int ChannelVector, int FeatureVector, int WindowRows,
          int WindowCols, typename ConvType>
struct FilterTile<T, ChannelVector, FeatureVector, WindowRows, WindowCols,
                  FilterFormat::HWCF, ConvType>
    : public helpers::RegisterTile3D<
          typename helpers::VectorType<T, FeatureVector>::type, WindowRows,
          WindowCols, ChannelVector> {
  static_assert(std::is_same_v<ConvType, conv_type::Forward> ||
                std::is_same_v<ConvType, conv_type::InputBackprop>);
  using VecType = typename helpers::VectorType<T, FeatureVector>::type;
  using helpers::RegisterTile3D<VecType, WindowRows, WindowCols,
                                ChannelVector>::data;

  template <typename Index, MULTI_PTR_TEMPLATE_DECL>
  SNN_ALWAYS_INLINE FilterTile(
      cl::sycl::multi_ptr<T const, MULTI_PTR_TEMPLATE> input,
      Index const offset, Index const n_channels, Index const n_features) {
    Index row_idx = offset;
    SNN_PRAGMA_UNROLL
    for (int i = 0; i < WindowRows; ++i) {
      Index col_idx = row_idx;
      SNN_PRAGMA_UNROLL
      for (int j = 0; j < WindowCols; ++j) {
        Index ch_idx = col_idx;
        SNN_PRAGMA_UNROLL
        for (int ch_v = 0; ch_v < ChannelVector; ++ch_v) {
          if constexpr(std::is_same_v<ConvType, conv_type::InputBackprop>){
            data(WindowRows - 1 - i, WindowCols - 1 - j, ch_v) =
              helpers::io::Load<VecType>()(input, ch_idx);
          }else{
            data(i, j, ch_v) = helpers::io::Load<VecType>()(input, ch_idx);
          }
          ch_idx += n_features;
        }
        col_idx += n_channels * n_features;
      }
      row_idx += WindowCols * n_channels * n_features;
    }
  }

};

/** FCHW: A 3D tile (SliceCount x WindowRows x WindowCols/ColVectorWidth)
 *  from the filter tensor, composed of sycl::vec<T,ColVectorWidth> elements.
 *  SliceCount is either FeatureCount (Forward) or ChannelCount (InputBackprop)
 *  and enables a thread to process multiple features xor channels. Note that since
 *  format is FCHW, load vectorization is done opportunistically along the row dimension
 *  if WindowCols is a multiple of a sycl::vec type.
 */
template <typename T, int ChannelCount, int FeatureCount, int WindowRows, int WindowCols, typename ConvType>
struct FilterTile<T, ChannelCount, FeatureCount, WindowRows, WindowCols,
                  FilterFormat::FCHW, ConvType>
    : public helpers::RegisterTile3D<T, std::is_same_v<ConvType, conv_type::Forward>
                                       ? FeatureCount
                                       : ChannelCount, WindowRows, WindowCols> {
 private:
  static_assert(std::is_same_v<ConvType, conv_type::Forward> ||
                std::is_same_v<ConvType, conv_type::InputBackprop>);
  static_assert(!std::is_same_v<ConvType, conv_type::Forward> ||
                ChannelCount == 1);
  static_assert(!std::is_same_v<ConvType, conv_type::InputBackprop> ||
                FeatureCount == 1);
  static constexpr int SliceCount = std::is_same_v<ConvType, conv_type::Forward>
                                       ? FeatureCount
                                       : ChannelCount;
  static constexpr int VectorWidth = helpers::io::get_vec_size(WindowCols);
  static_assert(WindowCols % VectorWidth == 0 && "Bad tile width/vector width combination.");
  static constexpr auto NumVecElems = WindowCols / VectorWidth;

  using VecType = typename helpers::VectorType<T, VectorWidth>::type;
 public:
  using helpers::RegisterTile3D<T, SliceCount, WindowRows, WindowCols>::data;

  template <typename Index, MULTI_PTR_TEMPLATE_DECL>
  SNN_ALWAYS_INLINE FilterTile(
      cl::sycl::multi_ptr<T const, MULTI_PTR_TEMPLATE> input,
      Index const offset, Index const n_channels) {
    Index slice_idx = offset;
    SNN_PRAGMA_UNROLL
    for (int slice = 0; slice < SliceCount; ++slice) {
      Index row_idx = slice_idx;
      SNN_PRAGMA_UNROLL
      for (int i = 0; i < WindowRows; ++i) {
        Index col_idx = row_idx;
        SNN_PRAGMA_UNROLL
        for (int j = 0; j < WindowCols; j+=VectorWidth) {
          VecType filter_vals = helpers::io::Load<VecType>()(input, col_idx);
          for (int v = 0; v < VectorWidth; ++v){
            if constexpr(std::is_same_v<ConvType, conv_type::InputBackprop>){
              data(slice, WindowRows - 1 - i, WindowCols - 1 - j - v) = 
                helpers::vector_element::get(filter_vals, v);
            }else{
              data(slice, i, j + v) =
                helpers::vector_element::get(filter_vals, v);
            }
          }
          col_idx += VectorWidth;
        }
        row_idx += WindowCols;
      }
      if constexpr(std::is_same_v<ConvType, conv_type::InputBackprop>){
        slice_idx += WindowRows * WindowCols; // Next channel
      }else{
        slice_idx += n_channels * WindowRows * WindowCols; // Next feature
      }

    }
  }
};

template <typename T, int VectorWidth, int OutTileRows, int OutTileCols, DataFormat Layout>
struct OutputTile;

/* NHWC: An OutTileRows x OutTileCols tile, composed of
 * sycl::vec<T,FeatureVectorWidth> elements, to collect output results.
 * FeatureVectorWidth allows multiple output features to be stored at once.
 */
template <typename T, int VectorWidth, int OutTileRows, int OutTileCols>
struct OutputTile<T, VectorWidth, OutTileRows, OutTileCols, DataFormat::NHWC> final
    : helpers::RegisterTile2D<
          typename helpers::VectorType<T, VectorWidth>::type, OutTileRows,
          OutTileCols> {
  using VecType = typename helpers::VectorType<T, VectorWidth>::type;
  using helpers::RegisterTile2D<VecType, OutTileRows, OutTileCols>::data;

  template <typename Index, MULTI_PTR_TEMPLATE_DECL>
  void SNN_ALWAYS_INLINE write_out(
      cl::sycl::multi_ptr<T, MULTI_PTR_TEMPLATE> output, Index const batch,
      Index const out_row, Index const n_rows, Index const out_col,
      Index const n_cols, Index const feature, Index const n_features) {
    // Check range bounds
    if (out_row + OutTileRows < n_rows && out_col + OutTileCols < n_cols) {
      write_out_impl<false>(output, batch, out_row, n_rows, out_col, n_cols,
                         feature, n_features);
    } else {
      write_out_impl<true>(output, batch, out_row, n_rows, out_col, n_cols,
                        feature, n_features);
    }
  }

 private:
  template <bool CheckRange, typename Index, MULTI_PTR_TEMPLATE_DECL>
  void SNN_ALWAYS_INLINE write_out_impl(
      cl::sycl::multi_ptr<T, MULTI_PTR_TEMPLATE> output, Index const batch,
      Index const out_row, Index const n_rows, Index const out_col,
      Index const n_cols, Index const feature, Index const n_features) {
    Index const offset =
        ((batch * n_rows + out_row) * n_cols + out_col) * n_features + feature;

    Index row_idx = offset;
    SNN_PRAGMA_UNROLL
    for (int tile_row = 0; tile_row < OutTileRows; ++tile_row) {
      if (!CheckRange || tile_row < n_rows - out_row) {
        Index idx = row_idx;
        SNN_PRAGMA_UNROLL
        for (int tile_col = 0; tile_col < OutTileCols; ++tile_col) {
          if (!CheckRange || tile_col < n_cols - out_col) {
            helpers::io::Store<VecType>()(output, idx,
                                          data(tile_row, tile_col));
            idx += n_features;
          }
        }
        row_idx += n_cols * n_features;
      }
    }
  }
};

/* NCHW: A 2D tile (OutTileRows x OutTileCols/VectorWidth), to collect output
 * results composed of T elements, to collect output results. VectorWidth allows
 * multiple values in a row to be stored at once.
 */
template <typename T, int OutFeatures, int OutTileRows, int OutTileCols>
struct OutputTile<T, OutFeatures, OutTileRows, OutTileCols, DataFormat::NCHW>
    final : helpers::RegisterTile3D<
                T, OutFeatures, OutTileRows,
                OutTileCols> {
 private:
  static constexpr int VectorWidth = helpers::io::get_vec_size(OutTileCols);
  static_assert(OutTileCols % VectorWidth == 0);
  static constexpr auto NumVecElems = OutTileCols / VectorWidth;
  using VecType = typename helpers::VectorType<T, VectorWidth>::type;
 public:
  using helpers::RegisterTile3D<T, OutFeatures, OutTileRows, OutTileCols>::data;

  template <typename Index, MULTI_PTR_TEMPLATE_DECL>
  void SNN_ALWAYS_INLINE write_out(
      cl::sycl::multi_ptr<T, MULTI_PTR_TEMPLATE> output, Index const batch,
      Index const out_row, Index const n_rows, Index const out_col,
      Index const n_cols, Index const feature, Index const n_features) {
    // Check range bounds
    if (out_row + OutTileRows < n_rows && out_col + OutTileCols < n_cols) {
      write_out_impl<false>(output, batch, out_row, n_rows, out_col, n_cols,
                         feature, n_features);
    } else {
      write_out_impl<true>(output, batch, out_row, n_rows, out_col, n_cols,
                        feature, n_features);
    }
  }

 private:
  template <bool CheckRange, typename Index, MULTI_PTR_TEMPLATE_DECL>
  void SNN_ALWAYS_INLINE write_out_impl(
      cl::sycl::multi_ptr<T, MULTI_PTR_TEMPLATE> output, Index const batch,
      Index const out_row, Index const n_rows, Index const out_col,
      Index const n_cols, Index const feature, Index const n_features) {

    Index const offset = batch * n_rows * n_cols * n_features +
                         feature * n_rows * n_cols + out_row * n_cols + out_col;

    Index feat_idx = offset;
    SNN_PRAGMA_UNROLL
    for (int tile_feature = 0; tile_feature < OutFeatures; ++tile_feature) {
      Index row_idx = feat_idx;
      SNN_PRAGMA_UNROLL
      for (int tile_row = 0; tile_row < OutTileRows; ++tile_row) {
        if (!CheckRange || tile_row < n_rows - out_row) {
          Index idx = row_idx;
          SNN_PRAGMA_UNROLL
          for (int i = 0; i < OutTileCols; ++i) {
            if (!CheckRange || i < n_cols - out_col) {
              helpers::io::Store<sycl::vec<T,1>>()(output, idx, data(tile_feature, tile_row, i));
              ++idx;
            }
          }
          row_idx += n_cols;
        }
      }
      feat_idx += n_rows * n_cols;
    }
  }
};

#undef MULTI_PTR_TEMPLATE_DECL
#undef MULTI_PTR_TEMPLATE

}  // namespace tiled
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_CONV2D_TILED_TILES_H_
