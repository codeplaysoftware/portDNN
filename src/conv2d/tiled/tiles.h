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
struct mirror_filter_tag {};

#if SNN_ENABLE_USM
#define MULTI_PTR_TEMPLATE_DECL          \
  cl::sycl::access::address_space Space, \
      cl::sycl::access::decorated DecorateAddress
#else
#define MULTI_PTR_TEMPLATE_DECL cl::sycl::access::address_space Space
#endif  // SNN_ENABLE_USM

#if SNN_ENABLE_USM
#define MULTI_PTR_TEMPLATE Space, DecorateAddress
#else
#define MULTI_PTR_TEMPLATE Space
#endif  // SNN_ENABLE_USM

// TODO(jtodd) can we rename ChannelVector to Vector & generalize it to
// vectorizing the smallest dimension?
// If we do that, the template params Width and VectorWidth are no longer
// independent
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

template <typename T, int VectorWidth, int Width>
struct InputRow<T, VectorWidth, Width, DataFormat::NCHW> final
    : public helpers::RegisterTile1D<
          typename helpers::VectorType<T, VectorWidth>::type, Width/VectorWidth> {
 public:
  static_assert(Width % VectorWidth == 0 && "Bad tile width/vector width combination.");
  using VecType = typename helpers::VectorType<T, VectorWidth>::type;
  static constexpr auto NumVecElems = Width / VectorWidth;
  using helpers::RegisterTile1D<VecType, NumVecElems>::data;
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
    for (int i = 0; i < NumVecElems; ++i) {
      data(i) = helpers::io::Load<VecType>()(input, idx);
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
    for (int i = 0; i < NumVecElems; ++i) {
      data(i) = (col + i < 0 || col + i >= n_cols)
                    ? VecType{0}
                    : helpers::io::Load<VecType>()(input, idx);
      idx += VectorWidth;
    }
  }
};

template <typename T, int ChannelVector, int FeatureVector, int WindowRows,
          int WindowCols, FilterFormat Layout>
struct FilterTile;

/** HWCF: A 3D tile (WindowRows x WindowCols x ChannelVector) from the filter tensor,
 * composed of sycl::vec<T,FeatureVector> elements.
*/
template <typename T, int ChannelVector, int FeatureVector, int WindowRows,
          int WindowCols>
struct FilterTile<T, ChannelVector, FeatureVector, WindowRows, WindowCols,
                  FilterFormat::HWCF>
    : public helpers::RegisterTile3D<
          typename helpers::VectorType<T, FeatureVector>::type, WindowRows,
          WindowCols, ChannelVector> {
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
          data(i, j, ch_v) = helpers::io::Load<VecType>()(input, ch_idx);
          ch_idx += n_features;
        }
        col_idx += n_channels * n_features;
      }
      row_idx += WindowCols * n_channels * n_features;
    }
  }

  template <typename Index, MULTI_PTR_TEMPLATE_DECL>
  SNN_ALWAYS_INLINE FilterTile(
      cl::sycl::multi_ptr<T const, MULTI_PTR_TEMPLATE> input,
      Index const offset, Index const n_channels, Index const n_features,
      mirror_filter_tag) {
    Index row_idx = offset;
    SNN_PRAGMA_UNROLL
    for (int i = 0; i < WindowRows; ++i) {
      Index col_idx = row_idx;
      SNN_PRAGMA_UNROLL
      for (int j = 0; j < WindowCols; ++j) {
        Index ch_idx = col_idx;
        SNN_PRAGMA_UNROLL
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

/** FCHW: A 3D tile (WindowRows x WindowCols/ColVectorWidth x 1)
 *  from the filter tensor, composed of sycl::vec<T,ColVectorWidth> elements.
 *  The redundant 3rd dimension remains in case, in future, we want the option
 *  to load  the filter for every channel at once.
 */
template <typename T, int ColVectorWidth, int WindowRows, int WindowCols>
struct FilterTile<T, 1, ColVectorWidth, WindowRows, WindowCols,
                  FilterFormat::FCHW>
    : public helpers::RegisterTile3D<
          typename helpers::VectorType<T, ColVectorWidth>::type, WindowRows,
          WindowCols / ColVectorWidth, 1> {
  static_assert(WindowCols % ColVectorWidth == 0 && "Bad tile width/vector width combination.");
  static constexpr auto NumVecElems = WindowCols / ColVectorWidth;

  using VecType = typename helpers::VectorType<T, ColVectorWidth>::type;
  using helpers::RegisterTile3D<VecType, WindowRows, NumVecElems,
                                1>::data;

  template <typename Index, MULTI_PTR_TEMPLATE_DECL>
  SNN_ALWAYS_INLINE FilterTile(
      cl::sycl::multi_ptr<T const, MULTI_PTR_TEMPLATE> input,
      Index const offset, Index const n_channels, Index const n_features) {
    SNN_UNUSED_VAR(n_channels);
    SNN_UNUSED_VAR(n_features);
    Index row_idx = offset;
    SNN_PRAGMA_UNROLL
    for (int i = 0; i < WindowRows; ++i) {
      Index col_idx = row_idx;
      SNN_PRAGMA_UNROLL
      for (int j = 0; j < NumVecElems; ++j) {
        data(i, j, 0) = helpers::io::Load<VecType>()(input, col_idx);
        col_idx += ColVectorWidth;
      }
      row_idx += WindowCols;
    }
  }

  template <typename Index, MULTI_PTR_TEMPLATE_DECL>
  SNN_ALWAYS_INLINE FilterTile(
      cl::sycl::multi_ptr<T const, MULTI_PTR_TEMPLATE> input,
      Index const offset, Index const n_channels, Index const n_features,
      mirror_filter_tag) {
    SNN_UNUSED_VAR(n_channels);
    SNN_UNUSED_VAR(n_features);
    Index row_idx = offset;
    SNN_PRAGMA_UNROLL
    for (int i = 0; i < WindowRows; ++i) {
      Index col_idx = row_idx;
      SNN_PRAGMA_UNROLL
      for (int j = 0; j < NumVecElems; ++j) {
          data(WindowRows - 1 - i, WindowCols - 1 - j, 0) =
              helpers::io::Load<VecType>()(input, col_idx);
        col_idx += ColVectorWidth;
      }
      row_idx += WindowCols;
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
    if (out_row + OutTileRows < n_rows && out_col + OutTileCols < n_cols) {
      write_out_no_check(output, batch, out_row, n_rows, out_col, n_cols,
                         feature, n_features);
    } else {
      write_out_checked(output, batch, out_row, n_rows, out_col, n_cols,
                        feature, n_features);
    }
  }

 private:
  template <typename Index, MULTI_PTR_TEMPLATE_DECL>
  void SNN_ALWAYS_INLINE write_out_checked(
      cl::sycl::multi_ptr<T, MULTI_PTR_TEMPLATE> output, Index const batch,
      Index const out_row, Index const n_rows, Index const out_col,
      Index const n_cols, Index const feature, Index const n_features) {
    Index const offset =
        ((batch * n_rows + out_row) * n_cols + out_col) * n_features + feature;

    Index row_idx = offset;
    SNN_PRAGMA_UNROLL
    for (int tile_row = 0; tile_row < OutTileRows; ++tile_row) {
      if (tile_row < n_rows - out_row) {
        Index idx = row_idx;
        SNN_PRAGMA_UNROLL
        for (int tile_col = 0; tile_col < OutTileCols; ++tile_col) {
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

  template <typename Index, MULTI_PTR_TEMPLATE_DECL>
  void SNN_ALWAYS_INLINE write_out_no_check(
      cl::sycl::multi_ptr<T, MULTI_PTR_TEMPLATE> output, Index const batch,
      Index const out_row, Index const n_rows, Index const out_col,
      Index const n_cols, Index const feature, Index const n_features) {
    Index const offset =
        ((batch * n_rows + out_row) * n_cols + out_col) * n_features + feature;

    Index row_idx = offset;
    SNN_PRAGMA_UNROLL
    for (int tile_row = 0; tile_row < OutTileRows; ++tile_row) {
      Index idx = row_idx;
      SNN_PRAGMA_UNROLL
      for (int tile_col = 0; tile_col < OutTileCols; ++tile_col) {
        helpers::io::Store<VecType>()(output, idx, data(tile_row, tile_col));
        idx += n_features;
      }
      row_idx += n_cols * n_features;
    }
  }
};

/* NCHW: A 2D tile (OutTileRows x OutTileCols/VectorWidth), to collect output
 * results composed of T elements, to collect output results. VectorWidth allows
 * multiple values in a row to be stored at once.
 */
template <typename T, int VectorWidth, int OutTileRows, int OutTileCols>
struct OutputTile<T, VectorWidth, OutTileRows, OutTileCols, DataFormat::NCHW>
    final : helpers::RegisterTile2D<
                typename helpers::VectorType<T, VectorWidth>::type, OutTileRows,
                OutTileCols / VectorWidth> {
  static_assert(OutTileCols % VectorWidth == 0);
  static constexpr auto NumVecElems = OutTileCols / VectorWidth;
  using VecType = typename helpers::VectorType<T, VectorWidth>::type;
  using helpers::RegisterTile2D<VecType, OutTileRows, NumVecElems>::data;

  template <typename Index, MULTI_PTR_TEMPLATE_DECL>
  void SNN_ALWAYS_INLINE write_out(
      cl::sycl::multi_ptr<T, MULTI_PTR_TEMPLATE> output, Index const batch,
      Index const out_row, Index const n_rows, Index const out_col,
      Index const n_cols, Index const feature, Index const n_features) {
    if (out_row + OutTileRows < n_rows && out_col + OutTileCols < n_cols) {
      write_out_no_check(output, batch, out_row, n_rows, out_col, n_cols,
                         feature, n_features);
    } else {
      write_out_checked(output, batch, out_row, n_rows, out_col, n_cols,
                        feature, n_features);
    }
  }

 private:
  template <typename Index, MULTI_PTR_TEMPLATE_DECL>
  void SNN_ALWAYS_INLINE write_out_checked(
      cl::sycl::multi_ptr<T, MULTI_PTR_TEMPLATE> output, Index const batch,
      Index const out_row, Index const n_rows, Index const out_col,
      Index const n_cols, Index const feature, Index const n_features) {

    Index const offset = batch * n_rows * n_cols * n_features +
                         feature * n_rows * n_cols + out_row * n_cols + out_col;

    Index row_idx = offset;
    SNN_PRAGMA_UNROLL
    for (int tile_row = 0; tile_row < OutTileRows; ++tile_row) {
      if (tile_row < n_rows - out_row) {
        Index idx = row_idx;
        SNN_PRAGMA_UNROLL
        for (int i = 0; i < NumVecElems; ++i) {
          if (i < n_cols - out_col) {
            helpers::io::Store<VecType>()(output, idx,
                                          data(tile_row, i));
            idx += VectorWidth;
          }
        }
        row_idx += n_cols;
      }
    }
  }

  template <typename Index, MULTI_PTR_TEMPLATE_DECL>
  void SNN_ALWAYS_INLINE write_out_no_check(
      cl::sycl::multi_ptr<T, MULTI_PTR_TEMPLATE> output, Index const batch,
      Index const out_row, Index const n_rows, Index const out_col,
      Index const n_cols, Index const feature, Index const n_features) {

    Index const offset = batch * n_rows * n_cols * n_features +
                         feature * n_rows * n_cols + out_row * n_cols + out_col;
    Index row_idx = offset;
    SNN_PRAGMA_UNROLL
    for (int tile_row = 0; tile_row < OutTileRows; ++tile_row) {
      Index idx = row_idx;
      SNN_PRAGMA_UNROLL
      for (int i = 0; i < NumVecElems; ++i) {
        helpers::io::Store<VecType>()(output, idx, data(tile_row, i));
        idx += VectorWidth;
      }
      row_idx += n_cols;
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
