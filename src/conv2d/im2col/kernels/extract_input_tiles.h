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
#ifndef PORTDNN_SRC_CONV2D_IM2COL_KERNELS_EXTRACT_INPUT_TILES_H_
#define PORTDNN_SRC_CONV2D_IM2COL_KERNELS_EXTRACT_INPUT_TILES_H_

#include "portdnn/accessor_types.h"

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/params.h"

#include "portdnn/helpers/macros.h"
#include "portdnn/helpers/minmax.h"

#include "src/helpers/tensor_index.h"
#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"
#include "src/helpers/window_index.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace im2col {

template <typename T, typename Index, int VectorWidth, typename ConvType,
          bool isUSM>
struct ExtractInputTiles;
/**
 * Have one thread per input entry. That thread is then responsible for writing
 * its one entry to each point in the intermediate tensor as required for the
 * contraction.
 */
template <typename T, typename Index, int VectorWidth, bool isUSM>
struct ExtractInputTiles<T, Index, VectorWidth, conv_type::Forward, isUSM> {
  using VecType = typename helpers::VectorType<T, VectorWidth>::type;
  using Load = helpers::io::Load<VecType>;
  using Store = helpers::io::Store<VecType>;

  ExtractInputTiles(Index tile_size, Conv2DParams const& params,
                    ReadMem<T const, isUSM> const& input,
                    WriteMem<T, isUSM> const& output)
      : tile_size_{params.group_format == sycldnn::BatchFormat::STRIDED
                       ? tile_size
                       : tile_size * params.groups},
        groups_{params.group_format == sycldnn::BatchFormat::STRIDED
                    ? params.groups
                    : 1},
        channels_{params.group_format == sycldnn::BatchFormat::STRIDED
                      ? params.channels / params.groups
                      : params.channels},
        features_{params.features},
        batch_{params.batch},
        in_rows_{params.in_rows},
        in_cols_{params.in_cols},
        window_rows_{params.window_rows},
        window_cols_{params.window_cols},
        stride_rows_{params.stride_rows},
        stride_cols_{params.stride_cols},
        out_rows_{params.out_rows},
        out_cols_{params.out_cols},
        pad_rows_{params.window_rows - params.pad_rows - 1},
        pad_cols_{params.window_cols - params.pad_cols - 1},
        input_accessor_{input},
        output_accessor_{output} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<3> item) const {
    Index const channel = item.get_id(0) * VectorWidth;
    Index const col_idx = item.get_id(1);
    Index row_idx;
    Index batch;
    if (batch_ == 1) {
      row_idx = item.get_id(2);
      batch = 0;
    } else {
      auto const tensor_idx =
          helpers::TensorIndexHelper<Index, false>::unflatten2d(
              item.get_id(2), in_rows_, in_rows_);
      row_idx = tensor_idx.s1;
      batch = tensor_idx.s0;
    }

    Index group;
    Index group_channel;
    if (groups_ == 1) {
      group = 0;
      group_channel = channel;
    } else {
      auto channel_groups_idx =
          helpers::TensorIndexHelper<Index, false>::unflatten2d(
              channel, channels_, channels_);

      group = channel_groups_idx.s0;
      group_channel = channel_groups_idx.s1;
    }

    if (group_channel < channels_ && group < groups_ && col_idx < in_cols_ &&
        row_idx < in_rows_ && batch < batch_) {
      auto input_data = input_accessor_.get_pointer();
      auto output_data = output_accessor_.get_pointer();

      Index const in_idx =
          (((batch * in_rows_ + row_idx) * in_cols_ + col_idx) * groups_ +
           group) *
              channels_ +
          group_channel;
      VecType in_val = Load()(input_data, in_idx);

      auto const col_window_struct =
          helpers::out_window_from_input(col_idx, stride_cols_, pad_cols_);
      Index const cstart = col_window_struct.window_start;
      Index const firstc = col_window_struct.filter_start;

      auto const row_window_struct =
          helpers::out_window_from_input(row_idx, stride_rows_, pad_rows_);
      Index const rstart = row_window_struct.window_start;
      Index const firstr = row_window_struct.filter_start;

      for (Index r = rstart, in_r = window_rows_ - 1 - firstr; in_r >= 0;
           ++r, in_r -= stride_rows_) {
        if (r >= 0 && r < out_rows_) {
          for (Index c = cstart, in_c = window_cols_ - 1 - firstc; in_c >= 0;
               ++c, in_c -= stride_cols_) {
            if (c >= 0 && c < out_cols_) {
              auto tile_start =
                  output_data +
                  (((group * batch_ + batch) * out_rows_ + r) * out_cols_ + c) *
                      tile_size_;
              Index tile_idx =
                  (in_r * window_cols_ + in_c) * channels_ + group_channel;
              Store()(tile_start, tile_idx, in_val);
            }
          }
        }
      }
    }
  }

 private:
  Index const tile_size_;
  Index const groups_;
  Index const channels_;
  Index const features_;
  Index const batch_;
  Index const in_rows_;
  Index const in_cols_;
  Index const window_rows_;
  Index const window_cols_;
  Index const stride_rows_;
  Index const stride_cols_;
  Index const out_rows_;
  Index const out_cols_;
  Index const pad_rows_;
  Index const pad_cols_;
  ReadMem<T const, isUSM> input_accessor_;
  WriteMem<T, isUSM> output_accessor_;
};

template <typename T, typename Index, int VectorWidth, bool isUSM>
struct ExtractInputTiles<T, Index, VectorWidth, conv_type::InputBackprop,
                         isUSM> {
  using VecType = typename helpers::VectorType<T, VectorWidth>::type;
  using Load = helpers::io::Load<VecType>;
  using Store = helpers::io::Store<VecType>;

  ExtractInputTiles(Index tile_size, Conv2DParams const& params,
                    ReadMem<T const, isUSM> const& input,
                    WriteMem<T, isUSM> const& output)
      : tile_size_{tile_size},
        channels_{params.channels},
        features_{params.features},
        batch_{params.batch},
        in_rows_{params.in_rows},
        in_cols_{params.in_cols},
        window_rows_{params.window_rows},
        window_cols_{params.window_cols},
        stride_rows_{params.stride_rows},
        stride_cols_{params.stride_cols},
        out_rows_{params.out_rows},
        out_cols_{params.out_cols},
        pad_rows_{params.pad_rows},
        pad_cols_{params.pad_cols},
        input_accessor_{input},
        output_accessor_{output} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<3> item) const {
    Index const feature = item.get_id(0) * VectorWidth;
    Index const col_idx = item.get_id(1);
    Index row_idx;
    Index batch;
    if (batch_ == 1) {
      row_idx = item.get_id(2);
      batch = 0;
    } else {
      auto const tensor_idx =
          helpers::TensorIndexHelper<Index, false>::unflatten2d(
              item.get_id(2), out_rows_, out_rows_);
      row_idx = tensor_idx.s1;
      batch = tensor_idx.s0;
    }
    if (feature < features_ && col_idx < out_cols_ && row_idx < out_rows_ &&
        batch < batch_) {
      auto input_data = input_accessor_.get_pointer();
      auto output_data = output_accessor_.get_pointer();

      Index const in_idx =
          ((batch * out_rows_ + row_idx) * out_cols_ + col_idx) * features_ +
          feature;
      VecType in_val = Load()(input_data, in_idx);

      Index const cstart = col_idx * stride_cols_ - pad_cols_;
      Index const rstart = row_idx * stride_rows_ - pad_rows_;

      for (Index r = rstart, in_r = window_rows_ - 1; in_r >= 0; ++r, --in_r) {
        if (r >= 0 && r < in_rows_) {
          for (Index c = cstart, in_c = window_cols_ - 1; in_c >= 0;
               ++c, --in_c) {
            if (c >= 0 && c < in_cols_) {
              auto tile_start =
                  output_data +
                  ((batch * in_rows_ + r) * in_cols_ + c) * tile_size_;
              Index tile_idx =
                  (in_r * window_cols_ + in_c) * features_ + feature;
              Store()(tile_start, tile_idx, in_val);
            }
          }
        }
      }
    }
  }

 private:
  Index const tile_size_;
  Index const channels_;
  Index const features_;
  Index const batch_;
  Index const in_rows_;
  Index const in_cols_;
  Index const window_rows_;
  Index const window_cols_;
  Index const stride_rows_;
  Index const stride_cols_;
  Index const out_rows_;
  Index const out_cols_;
  Index const pad_rows_;
  Index const pad_cols_;
  ReadMem<T const, isUSM> input_accessor_;
  WriteMem<T, isUSM> output_accessor_;
};

template <typename T, typename Index, int VectorWidth, bool isUSM>
struct ExtractInputTiles<T, Index, VectorWidth, conv_type::FilterBackprop,
                         isUSM> {
  using VecType = typename helpers::VectorType<T, 1>::type;
  using Load = helpers::io::Load<VecType>;
  using Store = helpers::io::Store<VecType>;

  ExtractInputTiles(Index tile_size, Conv2DParams const& params,
                    ReadMem<T const, isUSM> const& input,
                    WriteMem<T, isUSM> const& output)
      : tile_size_{tile_size},
        channels_{params.channels},
        features_{params.features},
        batch_{params.batch},
        in_rows_{params.in_rows},
        in_cols_{params.in_cols},
        window_rows_{params.window_rows},
        window_cols_{params.window_cols},
        stride_rows_{params.stride_rows},
        stride_cols_{params.stride_cols},
        out_rows_{params.out_rows},
        out_cols_{params.out_cols},
        pad_rows_{params.pad_rows},
        pad_cols_{params.pad_cols},
        dilation_rows_{params.dilation_rows},
        dilation_cols_{params.dilation_cols},
        input_accessor_{input},
        output_accessor_{output} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<3> item) const {
    Index const channel = item.get_id(0);
    Index const col_idx = item.get_id(1);
    Index row_idx;
    Index batch;
    if (batch_ == 1) {
      row_idx = item.get_id(2);
      batch = 0;
    } else {
      auto const tensor_idx =
          helpers::TensorIndexHelper<Index, false>::unflatten2d(
              item.get_id(2), in_rows_, in_rows_);
      row_idx = tensor_idx.s1;
      batch = tensor_idx.s0;
    }

    if (channel < channels_ && col_idx < in_cols_ && row_idx < in_rows_ &&
        batch < batch_) {
      auto input_data = input_accessor_.get_pointer();
      auto output_data = output_accessor_.get_pointer();

      Index const in_idx =
          ((batch * in_rows_ + row_idx) * in_cols_ + col_idx) * channels_ +
          channel;
      VecType in_val = Load()(input_data, in_idx);

      // c is the index in the padded output tensor (ie with lots of extra
      // zeros), but without the first padding. first_padded_c adds this extra
      // padding.
      Index const c = col_idx + pad_cols_;
      Index const first_padded_c = c - (window_cols_ - 1) * dilation_cols_;
      // The first and last output indices affected by this input.
      Index const last_used_c = c / stride_cols_;
      Index const cstart =
          helpers::round_ratio_up(first_padded_c, stride_cols_);
      Index const cend = helpers::min(last_used_c + 1, out_cols_);

      Index const r = row_idx + pad_rows_;
      Index const last_used_r = r / stride_rows_;
      Index const first_padded_r = r - (window_rows_ - 1) * dilation_rows_;
      Index const rstart =
          helpers::round_ratio_up(first_padded_r, stride_rows_);
      Index const rend = helpers::min(last_used_r + 1, out_rows_);

      Index init_r = rstart;
      Index init_r_idx = window_rows_ - 1;
      if (init_r < 0) {
        Index n_inc =
            helpers::round_ratio_up_above_zero(-init_r, dilation_rows_);
        init_r_idx -= n_inc * stride_rows_;
        init_r += n_inc * dilation_rows_;
      }
      Index init_c = cstart;
      Index init_c_idx = window_cols_ - 1;
      if (init_c < 0) {
        Index n_inc =
            helpers::round_ratio_up_above_zero(-init_c, dilation_cols_);
        init_c_idx -= n_inc * stride_cols_;
        init_c += n_inc * dilation_cols_;
      }

      for (Index r = init_r, in_r = init_r_idx; r < rend;
           r += dilation_rows_, in_r -= stride_rows_) {
        for (Index c = init_c, in_c = init_c_idx; c < cend;
             c += dilation_cols_, in_c -= stride_cols_) {
          auto tile_start =
              output_data +
              ((r * out_cols_ + c) * channels_ + channel) * tile_size_;
          Index tile_idx =
              ((batch * window_rows_ + in_r) * window_cols_ + in_c);
          Store()(tile_start, tile_idx, in_val);
        }
      }
    }
  }

 private:
  Index const tile_size_;
  Index const channels_;
  Index const features_;
  Index const batch_;
  Index const in_rows_;
  Index const in_cols_;
  Index const window_rows_;
  Index const window_cols_;
  Index const stride_rows_;
  Index const stride_cols_;
  Index const out_rows_;
  Index const out_cols_;
  Index const pad_rows_;
  Index const pad_cols_;
  Index const dilation_rows_;
  Index const dilation_cols_;
  ReadMem<T const, isUSM> input_accessor_;
  WriteMem<T, isUSM> output_accessor_;
};

}  // namespace im2col
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_SRC_CONV2D_IM2COL_KERNELS_EXTRACT_INPUT_TILES_H_
