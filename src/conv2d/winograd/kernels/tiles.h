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
#ifndef PORTDNN_SRC_CONV2D_WINOGRAD_KERNELS_TILES_H_
#define PORTDNN_SRC_CONV2D_WINOGRAD_KERNELS_TILES_H_

#include "portdnn/conv2d/conv_type.h"

#include "src/helpers/register_tile.h"
#include "src/helpers/vector_io.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace winograd {

template <typename Index>
struct SYCLOutputWindow {
  Index rsize;
  Index csize;
  Index offset;
};

template <typename T, int M, int N, int R, int S>
struct InputTile final
    : public helpers::RegisterTile2D<T, M + R - 1, N + S - 1> {
  static constexpr int A = M + R - 1;
  static constexpr int B = N + S - 1;
  using helpers::RegisterTile2D<T, A, B>::data;

  /**
   * Read the input data from the provided input array. The pointer is assumed
   * to be at the first value that should be read into the input tile.
   *
   * The input is expected to be in the NHWC data format.
   *
   * NOTE: The template here allows different address space attributes to be
   * passed with the pointer, rather than specifying the pointer will be to
   * global memory or to local memory.
   */
  template <typename PtrT, MULTI_PTR_TEMPLATE_DECL, typename Index>
  SNN_ALWAYS_INLINE InputTile(
      cl::sycl::multi_ptr<PtrT const, MULTI_PTR_TEMPLATE> input,
      Index const batch, Index const rstart, Index const n_rows,
      Index const cstart, Index const n_cols, Index const channel,
      Index const n_channels)
      : helpers::RegisterTile2D<T, A, B>{} {
    Index const offset =
        ((batch * n_rows + rstart) * n_cols + cstart) * n_channels + channel;
    input += offset;
    Index row_idx = 0;
    SNN_PRAGMA_UNROLL
    for (int r = 0; r < A; ++r) {
      if (r >= -rstart && r < n_rows - rstart) {
        Index idx = row_idx;
        SNN_PRAGMA_UNROLL
        for (int c = 0; c < B; ++c) {
          if (c >= -cstart && c < n_cols - cstart) {
            data(r, c) = helpers::io::Load<T>()(input, idx);
          }
          idx += n_channels;
        }
      }
      row_idx += n_cols * n_channels;
    }
  }
};

template <typename T, int M, int N, int R, int S>
using BaseFilterTile = helpers::RegisterTile2D<T, R, S>;

template <typename T, int M, int N, int R, int S, typename ConvType>
struct FilterTile;

template <typename T, int M, int N, int R, int S>
struct FilterTile<T, M, N, R, S, conv_type::Forward> final
    : public BaseFilterTile<T, M, N, R, S> {
  using BaseFilterTile<T, M, N, R, S>::data;
  /**
   * Read the filter data from the provided input array. The pointer is assumed
   * to be at the start of the filter tensor.
   *
   * The input is expected to be in (Height x Width x Channel x Feature) format.
   * The height of the filter (no. of rows) is expected to be R, and the width
   * (no. of cols) is S.
   *
   * NOTE: The template here allows different address space attributes to be
   * passed with the pointer, rather than specifying the pointer will be to
   * global memory or to local memory.
   */
  template <typename PtrT, MULTI_PTR_TEMPLATE_DECL, typename Index>
  SNN_ALWAYS_INLINE FilterTile(
      cl::sycl::multi_ptr<PtrT const, MULTI_PTR_TEMPLATE> input,
      Index const channel, Index const feature, Index const n_channels,
      Index const n_features) {
    input += channel * n_features + feature;
    SNN_PRAGMA_UNROLL
    for (int r = 0; r < R; ++r) {
      SNN_PRAGMA_UNROLL
      for (int c = 0; c < S; ++c) {
        Index idx = (r * S + c) * n_channels * n_features;
        data(r, c) = helpers::io::Load<T>()(input, idx);
      }
    }
  }
};

template <typename T, int M, int N, int R, int S>
struct FilterTile<T, M, N, R, S, conv_type::InputBackprop> final
    : public BaseFilterTile<T, M, N, R, S> {
  using BaseFilterTile<T, M, N, R, S>::data;
  /**
   * Read the filter data from the provided input array but mirror the filter
   * for use in backprop. The pointer is assumed to be at the start of the
   * filter tensor.
   *
   * The input is expected to be in (Height x Width x Channel x Feature) format.
   *
   * NOTE: The template here allows different address space attributes to be
   * passed with the pointer, rather than specifying the pointer will be to
   * global memory or to local memory.
   */
  template <typename PtrT, MULTI_PTR_TEMPLATE_DECL, typename Index>
  SNN_ALWAYS_INLINE FilterTile(
      cl::sycl::multi_ptr<PtrT const, MULTI_PTR_TEMPLATE> input,
      Index const channel, Index const feature, Index const n_channels,
      Index const n_features) {
    input += channel * n_features + feature;
    SNN_PRAGMA_UNROLL
    for (int r = 0; r < R; ++r) {
      SNN_PRAGMA_UNROLL
      for (int c = 0; c < S; ++c) {
        // Here the transforms (R - 1 - r) and (S - 1 - c) mirror the filter
        // data. Note that the channel and feature dims were switched in the
        // kernel params.
        Index idx = (r * S + c) * n_channels * n_features;
        data(R - 1 - r, S - 1 - c) = helpers::io::Load<T>()(input, idx);
      }
    }
  }
};

template <typename T, int M, int N, int R, int S>
struct FilterTile<T, M, N, R, S, conv_type::FilterBackprop> final
    : public BaseFilterTile<T, M, N, R, S> {
  using BaseFilterTile<T, M, N, R, S>::data;
  /**
   * Read the filter data from the provided input array.
   *
   * The input is expected to be in (Batch x Height x Width x Feature) format.
   *
   * NOTE: The template here allows different address space attributes to be
   * passed with the pointer, rather than specifying the pointer will be to
   * global memory or to local memory.
   */
  template <typename PtrT, MULTI_PTR_TEMPLATE_DECL, typename Index>
  SNN_ALWAYS_INLINE FilterTile(
      cl::sycl::multi_ptr<PtrT const, MULTI_PTR_TEMPLATE> input,
      SYCLOutputWindow<Index> const& w, Index const n_cols,
      Index const n_features) {
    input += w.offset;
    for (int r = 0; r < R; ++r) {
      for (int c = 0; c < S; ++c) {
        Index idx = (r * n_cols + c) * n_features;
        if (r >= w.rsize || c >= w.csize) {
          data(r, c) = static_cast<T>(0);
        } else {
          data(r, c) = helpers::io::Load<T>()(input, idx);
        }
      }
    }
  }
};

/**
 * Base class for the output tile which provides a correctly sized data array.
 */
template <typename T, int M, int N, int R, int S>
using BaseTransformedFilterTile =
    helpers::RegisterTile2D<T, M + R - 1, N + S - 1>;
/**
 * Tile which transforms the intermediate layer into the output layer. Should be
 * specialised for each Winograd transform.
 *
 * This object needs to provide the following constructor:
 *
 *   template <bool mirror>
 *   TransformedFilterTile(FilterTile<T, 2, 2, 3, 3, mirror> const& filter)
 *       : BaseTransformedFilterTile<T, 2, 2, 3, 3>{} {
 *     // Implement the filter Winograd transform
 *   }
 *
 * and needs to provide access to the base class data array with:
 *
 *   using BaseTransformedFilterTile<T, 2, 2, 3, 3>::data;
 */
template <typename T, int M, int N, int R, int S>
struct TransformedFilterTile;

/**
 * Base class for the output tile which provides a correctly sized data array.
 */
template <typename T, int M, int N, int R, int S>
using BaseTransformedInputTile =
    helpers::RegisterTile2D<T, M + R - 1, N + S - 1>;

/**
 * Tile which transforms the intermediate layer into the output layer. Should be
 * specialised for each Winograd transform.
 *
 * This object needs to provide the following constructor:
 *
 *   TransformedInputTile(InputTile<T, 2, 2, 3, 3> const& inp)
 *       : BaseTransformedInputTile<T, 2, 2, 3, 3>{} {
 *     // Implement the input Winograd transform
 *   }
 *
 * and needs to provide access to the base class data array with:
 *
 *   using BaseTransformedInpuTile<T, 2, 2, 3, 3>::data;
 */
template <typename T, int M, int N, int R, int S>
struct TransformedInputTile;

/**
 * Tile to store the intermediate Winograd data. Provides an update method to
 * increment the tile with provided transformed inputs and filters.
 */
template <typename T, int M, int N, int R, int S>
struct IntermediateTile final
    : public helpers::RegisterTile2D<T, M + R - 1, N + S - 1> {
  static constexpr int A = M + R - 1;
  static constexpr int B = N + S - 1;
  using helpers::RegisterTile2D<T, A, B>::data;
  /**
   * Read the intermediate tile from a temporary buffer. The input shape is
   * expected to be
   *   [ (M+R-1)(N+S-1), (batch * tile_rows * tile_cols), features ].
   */
  template <typename PtrT, MULTI_PTR_TEMPLATE_DECL, typename Index>
  SNN_ALWAYS_INLINE IntermediateTile(
      cl::sycl::multi_ptr<PtrT const, MULTI_PTR_TEMPLATE> input,
      Index const tile_idx, Index const n_tiles, Index const feature,
      Index const n_features) {
    input += tile_idx * n_features + feature;
    for (int r = 0; r < A; ++r) {
      for (int c = 0; c < B; ++c) {
        Index idx = (r * B + c) * n_features * n_tiles;
        data(r, c) = helpers::io::Load<T>()(input, idx);
      }
    }
  }
};

/**
 * Base class for the output tile which provides a correctly sized data array.
 */
template <typename T, int M, int N, int R, int S>
using BaseOutputTile = helpers::RegisterTile2D<T, M, N>;

/**
 * Tile which transforms the intermediate layer into the output layer. Should be
 * specialised for each Winograd transform.
 *
 * This object needs to provide the following constructor:
 *
 *   OutputTile(IntermediateTile<T, M, N, R, S> const& tile)
 *       : BaseOutputTile<T, M, N, R, S>{} {
 *     // Implement the inverse Winograd transform
 *   }
 *
 * and needs to provide access to the base class data array with:
 *
 *   using BaseOutputTile<T, M, N, R, S>::data;
 */
template <typename T, int M, int N, int R, int S>
struct OutputTile;

template <typename T, int M, int N, int R, int S>
struct OutputData {
  static constexpr int A = M + R - 1;
  static constexpr int B = N + S - 1;
  /**
   * Write the transformed input tile to a temporary buffer where each entry of
   * the tile is split into separate matrices. The output pointer should be at
   * the start of the temporary buffer.
   *
   * The resulting temporary buffer will be written as a batch of these
   * matrices, with a shape of
   *   [ (M+R-1)*(N+S-1), (batch * row_tiles * col_tiles), channels ].
   *
   * NOTE: The template here allows different address space attributes to be
   * passed with the pointer, rather than specifying the pointer will be to
   * global memory or to local memory.
   */
  template <typename PtrT, MULTI_PTR_TEMPLATE_DECL, typename Index>
  static SNN_ALWAYS_INLINE void write_transformed_input(
      cl::sycl::multi_ptr<PtrT, MULTI_PTR_TEMPLATE> output,
      Index const tile_idx, Index const channel, Index const n_tiles,
      Index const n_channels, TransformedInputTile<T, M, N, R, S> const& tile) {
    output += tile_idx * n_channels + channel;
    Index idx = 0;
    for (int r = 0; r < A; ++r) {
      for (int c = 0; c < B; ++c) {
        helpers::io::Store<T>()(output, idx, tile.data(r, c));
        idx += n_tiles * n_channels;
      }
    }
  }
  /**
   * Write the transformed filter tile to a temporary buffer where each entry of
   * the tile is split into separate matrices. The output pointer should be at
   * the start of the temporary buffer.
   *
   * The resulting temporary buffer will be written as a batch of these
   * matrices, with a shape of
   *   [ (M+R-1)*(N+S-1), features, channels ].
   *
   * NOTE: The template here allows different address space attributes to be
   * passed with the pointer, rather than specifying the pointer will be to
   * global memory or to local memory.
   */
  template <typename PtrT, MULTI_PTR_TEMPLATE_DECL, typename Index>
  static SNN_ALWAYS_INLINE void write_transformed_filter(
      cl::sycl::multi_ptr<PtrT, MULTI_PTR_TEMPLATE> output, Index const channel,
      Index const feature, Index const n_channels, Index const n_features,
      TransformedFilterTile<T, M, N, R, S> const& tile) {
    output += feature * n_channels + channel;
    for (int r = 0; r < A; ++r) {
      for (int c = 0; c < B; ++c) {
        Index idx = (r * B + c) * n_features * n_channels;
        helpers::io::Store<T>()(output, idx, tile.data(r, c));
      }
    }
  }
  /**
   * Write the output tile to the correct output memory. The output pointer
   * should be at the start of the output buffer. The resulting output shape is
   * NHWC.
   *
   * NOTE: The template here allows different address space attributes to be
   * passed with the pointer, rather than specifying the pointer will be to
   * global memory or to local memory.
   */
  template <typename PtrT, MULTI_PTR_TEMPLATE_DECL, typename Index>
  static SNN_ALWAYS_INLINE void write_output(
      cl::sycl::multi_ptr<PtrT, MULTI_PTR_TEMPLATE> output,
      SYCLOutputWindow<Index> const& window, Index const n_cols,
      Index const n_channels, OutputTile<T, M, N, R, S> const& tile) {
    output += window.offset;
    for (int r = 0; r < M && r < window.rsize; ++r) {
      for (int c = 0; c < N && c < window.csize; ++c) {
        Index idx = (r * n_cols + c) * n_channels;
        helpers::io::Store<T>()(output, idx, tile.data(r, c));
      }
    }
  }
  /**
   * Write the output tile to the correct output memory. The output pointer
   * should be at the start of the output buffer. The resulting output shape is
   * HWCF.
   *
   * The filter has size M x N when run in FilterBackprop mode, so we don't need
   * to check the bounds for writing to the output.
   *
   * NOTE: The template here allows different address space attributes to be
   * passed with the pointer, rather than specifying the pointer will be to
   * global memory or to local memory.
   */
  template <bool accumulate_output, typename PtrT, MULTI_PTR_TEMPLATE_DECL,
            typename Index>
  static SNN_ALWAYS_INLINE void write_filter_output(
      cl::sycl::multi_ptr<PtrT, MULTI_PTR_TEMPLATE> output, Index const channel,
      Index const feature, Index const n_channels, Index const n_features,
      OutputTile<T, M, N, R, S> const& tile) {
    output += channel * n_features + feature;
    for (int r = 0; r < M; ++r) {
      for (int c = 0; c < N; ++c) {
        Index idx = (r * N + c) * n_channels * n_features;
        auto out_data = tile.data(r, c);
        if (accumulate_output) {
          out_data += helpers::io::Load<T>()(output, idx);
        }
        helpers::io::Store<T>()(output, idx, out_data);
      }
    }
  }
};

}  // namespace winograd
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#include "tiles_impl.h"

#endif  // PORTDNN_SRC_CONV2D_WINOGRAD_KERNELS_TILES_H_
