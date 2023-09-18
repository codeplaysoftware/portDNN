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
#ifndef PORTDNN_SRC_CONV2D_DIRECT_KERNELS_NHWC_H_
#define PORTDNN_SRC_CONV2D_DIRECT_KERNELS_NHWC_H_

#include "src/conv2d/direct/kernels.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace direct {
template <typename T, typename Index, bool UseFastDiv, int StaticWindow,
          int StaticStride, int VectorWidth, bool isUSM>
struct DirectConv2D<T, Index, conv_type::Forward, UseFastDiv, StaticWindow,
                    StaticStride, VectorWidth, layout::NHWC, isUSM> {
  using IndexDivType = typename fast_div::IndexDiv<Index, UseFastDiv>::type;

  using ScalarType = T;
  using LoadScalar = helpers::io::Load<ScalarType>;

  using DataType = typename helpers::VectorType<T, VectorWidth>::type;
  using LoadData = helpers::io::Load<DataType>;
  using StoreData = helpers::io::Store<DataType>;

  DirectConv2D(const Conv2DParams& params, const ReadMem<const T, isUSM> input,
               const ReadMem<const T, isUSM> filter, WriteMem<T, isUSM> output)
      : n_elems_{params.batch * params.out_rows * params.out_cols *
                 params.features / VectorWidth},
        div_features_{params.features / VectorWidth},
        div_out_cols_{params.out_cols},
        div_out_rows_{params.out_rows},
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
        input_mem_{input},
        filter_mem_{filter},
        output_mem_{output} {}

  inline SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) const {
    Index index = item.get_id(0);
    const Index range = item.get_range().get(0);

    for (; index < n_elems_; index += range) {
      const auto input_data = input_mem_.get_pointer();
      const auto filter_data = filter_mem_.get_pointer();
      auto output_data = output_mem_.get_pointer();

      const auto tensor_idx =
          helpers::TensorIndexHelper<Index, UseFastDiv>::unflatten4d(
              index, div_out_rows_, out_rows_, div_out_cols_, out_cols_,
              div_features_, features_ / VectorWidth);
      const Index feature = tensor_idx.s3 * VectorWidth;
      const Index col_idx = tensor_idx.s2;
      const Index row_idx = tensor_idx.s1;
      const Index batch = tensor_idx.s0;

      const Index col_stride = static_stride_param(stride_cols_);
      const auto col_window_struct =
          helpers::in_window_from_output(col_idx, col_stride, pad_cols_);
      const Index cstart = col_window_struct.window_start;
      const Index firstc = col_window_struct.filter_start;

      const Index row_stride = static_stride_param(stride_rows_);
      const auto row_window_struct =
          helpers::in_window_from_output(row_idx, row_stride, pad_rows_);
      const Index rstart = row_window_struct.window_start;
      const Index firstr = row_window_struct.filter_start;

      DataType out_val{0};

      const auto input_data_n =
          input_data + batch * in_cols_ * in_rows_ * channels_;
      const auto filter_data_n = filter_data + feature;
      const Index row_window = static_window_param(window_rows_);
      const Index col_window = static_window_param(window_cols_);

      Index in_row_idx = rstart * in_cols_ * channels_;
      Index fil_row_idx = firstr * col_window * channels_ * features_;
      for (Index r = rstart, i = firstr; i < row_window; ++r, ++i,
                 in_row_idx += in_cols_ * channels_,
                 fil_row_idx += col_window * channels_ * features_) {
        if (r >= 0 && r < in_rows_) {
          Index in_col_idx = in_row_idx + cstart * channels_;
          Index fil_col_idx = fil_row_idx + firstc * channels_ * features_;

          for (Index c = cstart, j = firstc; j < col_window; ++c, ++j,
                     in_col_idx += channels_,
                     fil_col_idx += channels_ * features_) {
            if (c >= 0 && c < in_cols_) {
              Index idx = in_col_idx;
              Index k_idx = fil_col_idx;

              for (Index channel = 0; channel < channels_;
                   ++channel, ++idx, k_idx += features_) {
                DataType in_val = DataType{LoadScalar()(input_data_n, idx)};
                DataType fil_vals = LoadData()(filter_data_n, k_idx);

                out_val = helpers::math::mad(in_val, fil_vals, out_val);
              }  // channel loop
            }
          }  // col loop
        }
      }  // row loop

      StoreData()(output_data, index * VectorWidth, out_val);
    }
  }

 private:
  /** Check whether the window size is available at compile time or whether the
   * runtime value has to be used. */
  constexpr Index static_window_param(Index window) const {
    return (StaticWindow > 0 ? StaticWindow : window);
  }
  /** Check whether the stride size is available at compile time or whether the
   * runtime value has to be used. */
  constexpr Index static_stride_param(Index stride) const {
    return (StaticStride > 0 ? StaticStride : stride);
  }

  const Index n_elems_;
  const IndexDivType div_features_;
  const IndexDivType div_out_cols_;
  const IndexDivType div_out_rows_;
  const Index channels_;
  const Index features_;
  const Index batch_;
  const Index in_rows_;
  const Index in_cols_;
  const Index window_rows_;
  const Index window_cols_;
  const Index stride_rows_;
  const Index stride_cols_;
  const Index out_rows_;
  const Index out_cols_;
  const Index pad_rows_;
  const Index pad_cols_;
  const ReadMem<const T, isUSM> input_mem_;
  const ReadMem<const T, isUSM> filter_mem_;
  WriteMem<T, isUSM> output_mem_;
};
template <typename T, typename Index, bool UseFastDiv, int StaticWindow,
          int StaticStride, int VectorWidth, bool isUSM>
struct DirectConv2D<T, Index, conv_type::InputBackprop, UseFastDiv,
                    StaticWindow, StaticStride, VectorWidth, layout::NHWC,
                    isUSM> {
  using IndexDivType = typename fast_div::IndexDiv<Index, UseFastDiv>::type;

  using ScalarType = T;
  using LoadScalar = helpers::io::Load<ScalarType>;
  using StoreScalar = helpers::io::Store<ScalarType>;

  using DataType = typename helpers::VectorType<T, VectorWidth>::type;
  using LoadData = helpers::io::Load<DataType>;
  using StoreData = helpers::io::Store<DataType>;

  DirectConv2D(const Conv2DParams& params, const ReadMem<const T, isUSM> input,
               const ReadMem<const T, isUSM> filter, WriteMem<T, isUSM> output)
      : n_elems_{params.batch * params.in_rows * params.in_cols *
                 params.features},
        div_features_{params.features},
        div_in_cols_{params.in_cols},
        div_in_rows_{params.in_rows},
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
        pad_rows_{static_window_param(params.window_rows) - params.pad_rows -
                  1},
        pad_cols_{static_window_param(params.window_cols) - params.pad_cols -
                  1},
        input_mem_{input},
        filter_mem_{filter},
        output_mem_{output} {}

  inline SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) const {
    Index index = item.get_id(0);
    const Index range = item.get_range().get(0);

    for (; index < n_elems_; index += range) {
      const auto input_data = input_mem_.get_pointer();
      const auto filter_data = filter_mem_.get_pointer();
      auto output_data = output_mem_.get_pointer();

      const auto tensor_idx =
          helpers::TensorIndexHelper<Index, UseFastDiv>::unflatten4d(
              index, div_in_rows_, in_rows_, div_in_cols_, in_cols_,
              div_features_, features_);
      const Index feature = tensor_idx.s3;
      const Index col_idx = tensor_idx.s2;
      const Index row_idx = tensor_idx.s1;
      const Index batch = tensor_idx.s0;

      const Index col_stride = static_stride_param(stride_cols_);
      const auto col_window_struct =
          helpers::out_window_from_input(col_idx, col_stride, pad_cols_);
      const Index cstart = col_window_struct.window_start;
      const Index firstc = col_window_struct.filter_start;

      const Index row_stride = static_stride_param(stride_rows_);
      const auto row_window_struct =
          helpers::out_window_from_input(row_idx, row_stride, pad_rows_);
      const Index rstart = row_window_struct.window_start;
      const Index firstr = row_window_struct.filter_start;

      ScalarType out_val{0};

      const auto input_data_n =
          input_data + batch * out_cols_ * out_rows_ * channels_;
      const auto filter_data_n = filter_data + feature * channels_;
      const Index row_window = static_window_param(window_rows_);
      const Index col_window = static_window_param(window_cols_);

      Index in_row_idx = rstart * out_cols_ * channels_;
      Index fil_row_idx =
          (row_window - firstr - 1) * col_window * features_ * channels_;
      for (Index r = rstart, i = firstr; i < row_window; ++r, i += row_stride,
                 in_row_idx += out_cols_ * channels_,
                 fil_row_idx -= row_stride * col_window * features_ *
                                                         channels_) {
        if (r >= 0 && r < out_rows_) {
          Index in_col_idx = in_row_idx + cstart * channels_;
          Index fil_col_idx =
              fil_row_idx + (col_window - firstc - 1) * features_ * channels_;

          for (Index c = cstart, j = firstc; j < col_window; ++c,
                     j += col_stride, in_col_idx += channels_,
                     fil_col_idx -= col_stride * features_ * channels_) {
            if (c >= 0 && c < out_cols_) {
              Index idx = in_col_idx;
              Index k_idx = fil_col_idx;

              for (Index channel = 0; channel < channels_;
                   channel += VectorWidth, idx += VectorWidth,
                         k_idx += VectorWidth) {
                DataType in_val = LoadData()(input_data_n, idx);
                DataType fil_val = LoadData()(filter_data_n, k_idx);

                out_val += helpers::math::dot(in_val, fil_val);
              }  // channel loop
            }
          }  // col loop
        }
      }  // row loop

      StoreScalar()(output_data, index, out_val);
    }
  }

 private:
  constexpr Index static_window_param(Index window) const {
    return (StaticWindow > 0 ? StaticWindow : window);
  }
  constexpr Index static_stride_param(Index stride) const {
    return (StaticStride > 0 ? StaticStride : stride);
  }

  const Index n_elems_;
  const IndexDivType div_features_;
  const IndexDivType div_in_cols_;
  const IndexDivType div_in_rows_;
  const Index channels_;
  const Index features_;
  const Index batch_;
  const Index in_rows_;
  const Index in_cols_;
  const Index window_rows_;
  const Index window_cols_;
  const Index stride_rows_;
  const Index stride_cols_;
  const Index out_rows_;
  const Index out_cols_;
  const Index pad_rows_;
  const Index pad_cols_;
  const ReadMem<const T, isUSM> input_mem_;
  const ReadMem<const T, isUSM> filter_mem_;
  WriteMem<T, isUSM> output_mem_;
};
/*
 * The main difference between the two backprop kernels is the way strides are
 * handled. In the filter backprop the input is strided and the filter is not
 * whereas in the input backprop this is the other way around.
 *
 * For the filter backprop we are convolving the input with the output as the
 * filter. This means that the static window sizes are actually the
 * params.out_rows_ and params.out_cols_ rather than the params.window_*.
 */
template <typename T, typename Index, bool UseFastDiv, int StaticOut,
          int StaticStride, int VectorWidth, bool isUSM>
struct DirectConv2D<T, Index, conv_type::FilterBackprop, UseFastDiv, StaticOut,
                    StaticStride, VectorWidth, layout::NHWC, isUSM> {
  using IndexDivType = typename fast_div::IndexDiv<Index, UseFastDiv>::type;

  using ScalarType = T;
  using LoadScalar = helpers::io::Load<ScalarType>;
  using StoreScalar = helpers::io::Store<ScalarType>;

  using DataType = typename helpers::VectorType<T, VectorWidth>::type;
  using LoadData = helpers::io::Load<DataType>;
  using StoreData = helpers::io::Store<DataType>;

  DirectConv2D(const Conv2DParams& params, const ReadMem<const T, isUSM> input,
               const ReadMem<const T, isUSM> filter, WriteMem<T, isUSM> output)
      : n_elems_{params.out_rows * params.out_cols * params.channels *
                 params.features / VectorWidth},
        div_features_{params.features / VectorWidth},
        div_channels_{params.channels},
        div_out_cols_{params.out_cols},
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
        input_mem_{input},
        filter_mem_{filter},
        output_mem_{output} {}

  inline SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) const {
    Index index = item.get_id(0);
    const Index range = item.get_range().get(0);

    for (; index < n_elems_; index += range) {
      const auto input_data = input_mem_.get_pointer();
      const auto filter_data = filter_mem_.get_pointer();
      auto output_data = output_mem_.get_pointer();

      const Index col_out = static_out_param(out_cols_);
      const auto tensor_idx =
          helpers::TensorIndexHelper<Index, UseFastDiv>::unflatten4d(
              index, div_out_cols_, col_out, div_channels_, channels_,
              div_features_, features_ / VectorWidth);
      const Index feature = tensor_idx.s3 * VectorWidth;
      const Index channel = tensor_idx.s2;
      const Index col_idx = tensor_idx.s1;
      const Index row_idx = tensor_idx.s0;

      const Index cstart = col_idx - pad_cols_;
      const Index cend = cstart + window_cols_;
      const Index rstart = row_idx - pad_rows_;
      const Index rend = rstart + window_rows_;

      const Index row_stride = static_stride_param(stride_rows_);
      const Index filter_rows =
          helpers::round_ratio_up_above_zero(window_rows_, row_stride);

      const Index col_stride = static_stride_param(stride_cols_);
      const Index filter_cols =
          helpers::round_ratio_up_above_zero(window_cols_, col_stride);

      DataType out_val{0};

      auto input_data_n = input_data + channel;
      auto filter_data_n = filter_data + feature;

      for (Index b = 0; b < batch_; b++) {
        Index in_row_idx = rstart * in_cols_ * channels_;
        Index fil_row_idx = 0;
        for (Index r = rstart; r < rend; r += row_stride,
                   in_row_idx += row_stride * in_cols_ * channels_,
                   fil_row_idx += filter_cols * features_) {
          if (r >= 0 && r < in_rows_) {
            Index idx = in_row_idx + cstart * channels_;
            Index k_idx = fil_row_idx;
            for (Index c = cstart; c < cend; c += col_stride,
                       idx += col_stride * channels_, k_idx += features_) {
              if (c >= 0 && c < in_cols_) {
                DataType in_val = DataType{LoadScalar()(input_data_n, idx)};
                DataType fil_vals = LoadData()(filter_data_n, k_idx);

                out_val = helpers::math::mad(in_val, fil_vals, out_val);
              }
            }  // col loop
          }
        }  // row loop

        input_data_n += in_cols_ * in_rows_ * channels_;
        filter_data_n += filter_rows * filter_cols * features_;
      }  // batch loop

      StoreData()(output_data, index * VectorWidth, out_val);
    }
  }

 private:
  constexpr Index static_out_param(Index out) const {
    return (StaticOut > 0 ? StaticOut : out);
  }
  constexpr Index static_stride_param(Index stride) const {
    return (StaticStride > 0 ? StaticStride : stride);
  }

  const Index n_elems_;
  const IndexDivType div_features_;
  const IndexDivType div_channels_;
  const IndexDivType div_out_cols_;
  const Index channels_;
  const Index features_;
  const Index batch_;
  const Index in_rows_;
  const Index in_cols_;
  const Index window_rows_;
  const Index window_cols_;
  const Index stride_rows_;
  const Index stride_cols_;
  const Index out_rows_;
  const Index out_cols_;
  const Index pad_rows_;
  const Index pad_cols_;
  const ReadMem<const T, isUSM> input_mem_;
  const ReadMem<const T, isUSM> filter_mem_;
  WriteMem<T, isUSM> output_mem_;
};

}  // namespace direct
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_SRC_CONV2D_DIRECT_KERNELS_NHWC_H_
