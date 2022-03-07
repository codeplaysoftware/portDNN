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
#ifndef SYCLDNN_SRC_CONV2D_DIRECT_KERNELS_NCHW_H_
#define SYCLDNN_SRC_CONV2D_DIRECT_KERNELS_NCHW_H_

#include "src/conv2d/direct/kernels.h"
#include "src/helpers/vector_element.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace direct {
template <typename T, typename Index, bool UseFastDiv, int StaticWindow,
          int StaticStride>
struct DirectConv2D<T, Index, conv_type::Forward, UseFastDiv, StaticWindow,
                    StaticStride, /*VectorWidth*/ 1, layout::NCHW> {
  using IndexDivType = typename fast_div::IndexDiv<Index, UseFastDiv>::type;

  DirectConv2D(const Conv2DParams& params, const ReadAccessor<const T> input,
               const ReadAccessor<const T> filter, WriteAccessor<T> output)
      : n_elems_{params.batch * params.out_rows * params.out_cols *
                 params.features},
        div_features_{params.features},
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
        input_accessor_{input},
        filter_accessor_{filter},
        output_accessor_{output} {}

  inline SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get_id(0);
    const Index range = item.get_range().get(0);

    for (; index < n_elems_; index += range) {
      const auto input_data = input_accessor_.get_pointer().get();
      const auto filter_data = filter_accessor_.get_pointer().get();
      auto output_data = output_accessor_.get_pointer().get();

      const auto tensor_idx =
          helpers::TensorIndexHelper<Index, UseFastDiv>::unflatten4d(
              index, div_features_, features_, div_out_rows_, out_rows_,
              div_out_cols_, out_cols_);
      const Index col_idx = tensor_idx.s3;
      const Index row_idx = tensor_idx.s2;
      const Index feature = tensor_idx.s1;
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

      T out_val{0};

      const Index row_window = static_window_param(window_rows_);
      const Index col_window = static_window_param(window_cols_);
      const auto input_data_n =
          input_data + batch * channels_ * in_rows_ * in_cols_;
      const auto filter_data_n =
          filter_data + feature * channels_ * row_window * col_window;

      for (Index channel = 0, in_chan_idx = 0, fil_chan_idx = 0;
           channel < channels_; ++channel, in_chan_idx += in_rows_ * in_cols_,
                 fil_chan_idx += row_window * col_window) {
        Index in_row_idx = in_chan_idx + rstart * in_cols_;
        Index fil_row_idx = fil_chan_idx + firstr * col_window;
        for (Index r = rstart, i = firstr; i < row_window;
             ++r, ++i, in_row_idx += in_cols_, fil_row_idx += col_window) {
          if (r >= 0 && r < in_rows_) {
            Index in_col_idx = in_row_idx + cstart;
            Index fil_col_idx = fil_row_idx + firstc;

            for (Index c = cstart, j = firstc; j < col_window;
                 ++c, ++j, ++in_col_idx, ++fil_col_idx) {
              if (c >= 0 && c < in_cols_) {
                T in_val = input_data_n[in_col_idx];
                T fil_val = filter_data_n[fil_col_idx];
                out_val = helpers::math::mad(in_val, fil_val, out_val);
              }
            }  // col loop
          }
        }  // row loop
      }    // channel loop

      output_data[index] = out_val;
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
  const ReadAccessor<const T> input_accessor_;
  const ReadAccessor<const T> filter_accessor_;
  WriteAccessor<T> output_accessor_;
};

}  // namespace direct
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_CONV2D_DIRECT_KERNELS_NCHW_H_