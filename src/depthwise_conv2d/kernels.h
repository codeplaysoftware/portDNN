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
#ifndef PORTDNN_SRC_DEPTHWISE_CONV2D_KERNELS_H_
#define PORTDNN_SRC_DEPTHWISE_CONV2D_KERNELS_H_

#include "portdnn/accessor_types.h"
#include "portdnn/helpers/macros.h"

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/depthwise_conv2d/params.h"

#include "src/helpers/math.h"
#include "src/helpers/tensor_index.h"
#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"
#include "src/helpers/window_index.h"
#include "src/helpers/workgroup_reduce.h"

namespace sycldnn {
namespace depthwise_conv2d {
namespace internal {

template <typename T, typename Index, typename ConvType, int VectorWidth,
          bool IsUSM>
struct DepthwiseConv2D;

template <typename T, typename Index, int VectorWidth, bool IsUSM>
struct DepthwiseConv2D<T, Index, conv2d::conv_type::Forward, VectorWidth,
                       IsUSM> {
  using DataType = typename helpers::VectorType<T, VectorWidth>::type;
  using Load = typename helpers::io::Load<DataType>;
  using Store = typename helpers::io::Store<DataType>;

  DepthwiseConv2D(Index n_elems, DepthwiseConv2DParams const& params,
                  ReadMem<T const, IsUSM> const& input,
                  ReadMem<T const, IsUSM> const& filter,
                  WriteMem<T, IsUSM> const& output)
      : n_elems_{n_elems / VectorWidth},
        features_{params.channels * params.channel_multiplier},
        p_{params},
        input_mem_{input},
        filter_mem_{filter},
        output_mem_{output} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<1> item) const {
    const Index index = item.get_id(0);

    if (index < n_elems_) {
      auto const input_data = input_mem_.get_pointer();
      auto const filter_data = filter_mem_.get_pointer();

      auto const tensor_idx =
          helpers::TensorIndexHelper<Index, false>::unflatten4d(
              index, p_.out_rows, p_.out_rows, p_.out_cols, p_.out_cols,
              features_ / VectorWidth, features_ / VectorWidth);
      Index const feature = tensor_idx.s3 * VectorWidth;
      Index const col_idx = tensor_idx.s2;
      Index const row_idx = tensor_idx.s1;
      Index const batch_idx = tensor_idx.s0;

      auto const feature_idx =
          helpers::TensorIndexHelper<Index, false>::unflatten2d(
              feature, p_.channel_multiplier, p_.channel_multiplier);
      Index const multiple = feature_idx.s1;
      Index const channel = feature_idx.s0;

      auto const col_window_struct =
          helpers::in_window_from_output(col_idx, p_.stride_cols, p_.pad_cols);
      Index const cstart = col_window_struct.window_start;
      Index const firstc = col_window_struct.filter_start;

      auto const row_window_struct =
          helpers::in_window_from_output(row_idx, p_.stride_rows, p_.pad_rows);
      Index const rstart = row_window_struct.window_start;
      Index const firstr = row_window_struct.filter_start;

      DataType out_val{0};
      Index const input_initial_offset =
          batch_idx * p_.in_cols * p_.in_rows * p_.channels + channel;
      Index const filter_initial_offset =
          channel * p_.channel_multiplier + multiple;

      Index input_row_offset =
          input_initial_offset + rstart * p_.in_cols * p_.channels;
      Index filter_row_offset =
          filter_initial_offset + firstr * p_.window_rows * features_;
      for (Index row = rstart, i = firstr; i < p_.window_rows; ++row, ++i) {
        if (row >= 0 && row < p_.in_rows) {
          Index input_offset = input_row_offset + cstart * p_.channels;
          Index filter_offset = filter_row_offset + firstc * features_;

          for (Index col = cstart, j = firstc; j < p_.window_cols; ++col, ++j) {
            if (col >= 0 && col < p_.in_cols) {
              DataType in_val = Load()(input_data, input_offset);
              DataType fil_val = Load()(filter_data, filter_offset);

              out_val = helpers::math::mad(in_val, fil_val, out_val);
            }

            input_offset += p_.channels;
            filter_offset += features_;
          }  // col loop
        }

        input_row_offset += p_.in_cols * p_.channels;
        filter_row_offset += p_.window_cols * features_;
      }  // row loop

      auto output_data = output_mem_.get_pointer();
      Store()(output_data, index * VectorWidth, out_val);
    }
  }

 private:
  Index const n_elems_;
  Index const features_;
  DepthwiseConv2DParams const p_;
  ReadMem<T const, IsUSM> const input_mem_;
  ReadMem<T const, IsUSM> const filter_mem_;
  WriteMem<T, IsUSM> output_mem_;
};

template <typename T, typename Index, int VectorWidth, bool IsUSM>
struct DepthwiseConv2D<T, Index, conv2d::conv_type::InputBackprop, VectorWidth,
                       IsUSM> {
  using DataType = typename helpers::VectorType<T, VectorWidth>::type;
  using Load = typename helpers::io::Load<DataType>;
  using Store = typename helpers::io::Store<DataType>;

  DepthwiseConv2D(Index n_elems, DepthwiseConv2DParams const& params,
                  ReadMem<T const, IsUSM> const& input,
                  ReadMem<T const, IsUSM> const& filter,
                  WriteMem<T, IsUSM> const& output)
      : n_elems_{n_elems / VectorWidth},
        features_{params.channels * params.channel_multiplier},
        p_{params},
        error_mem_{input},
        filter_mem_{filter},
        output_mem_{output} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<1> item) const {
    Index const index = item.get_id(0);

    if (index < n_elems_) {
      auto const input_data = error_mem_.get_pointer();
      auto const filter_data = filter_mem_.get_pointer();

      auto const tensor_idx =
          helpers::TensorIndexHelper<Index, false>::unflatten4d(
              index, p_.in_rows, p_.in_rows, p_.in_cols, p_.in_cols,
              p_.channels / VectorWidth, p_.channels / VectorWidth);
      Index const channel = tensor_idx.s3 * VectorWidth;
      Index const col_idx = tensor_idx.s2;
      Index const row_idx = tensor_idx.s1;
      Index const batch_idx = tensor_idx.s0;

      auto const col_window_struct =
          helpers::out_window_from_input(col_idx, p_.stride_cols, p_.pad_cols);
      Index const cstart = col_window_struct.window_start;
      Index const firstc = col_window_struct.filter_start;

      auto const row_window_struct =
          helpers::out_window_from_input(row_idx, p_.stride_rows, p_.pad_rows);
      Index const rstart = row_window_struct.window_start;
      Index const firstr = row_window_struct.filter_start;

      DataType out_val{0};
      Index const input_initial_offset =
          batch_idx * p_.out_cols * p_.out_rows * features_ +
          channel * p_.channel_multiplier;
      Index const filter_initial_offset = channel * p_.channel_multiplier;

      Index input_row_offset =
          input_initial_offset + rstart * p_.out_cols * features_;
      Index filter_row_offset =
          filter_initial_offset +
          (p_.window_rows - firstr - 1) * p_.window_cols * features_;
      for (Index row = rstart, i = firstr; i < p_.window_rows;
           ++row, i += p_.stride_rows) {
        if (row >= 0 && row < p_.out_rows) {
          Index input_col_offset = input_row_offset + cstart * features_;
          Index filter_col_offset =
              filter_row_offset + (p_.window_cols - firstc - 1) * features_;

          for (Index col = cstart, j = firstc; j < p_.window_cols;
               ++col, j += p_.stride_cols) {
            if (col >= 0 && col < p_.out_cols) {
              for (Index multiple = 0; multiple < p_.channel_multiplier;
                   ++multiple) {
                Index const idx = input_col_offset + multiple;
                DataType in_val = Load()(input_data, idx);

                Index const k_idx = filter_col_offset + multiple;
                DataType fil_val = Load()(filter_data, k_idx);

                out_val = helpers::math::mad(in_val, fil_val, out_val);
              }  // multiple loop
            }

            input_col_offset += features_;
            filter_col_offset -= p_.stride_cols * features_;
          }  // col loop
        }

        input_row_offset += p_.out_cols * features_;
        filter_row_offset -= p_.stride_rows * p_.window_cols * features_;
      }  // row loop

      auto output_data = output_mem_.get_pointer();
      Store()(output_data, index * VectorWidth, out_val);
    }
  }

 private:
  Index const n_elems_;
  Index const features_;
  DepthwiseConv2DParams const p_;
  ReadMem<T const, IsUSM> const error_mem_;
  ReadMem<T const, IsUSM> const filter_mem_;
  WriteMem<T, IsUSM> output_mem_;
};

template <typename T, typename Index, int VectorWidth, bool IsUSM>
struct DepthwiseConv2D<T, Index, conv2d::conv_type::FilterBackprop, VectorWidth,
                       IsUSM> {
  using DataType = typename helpers::VectorType<T, VectorWidth>::type;
  using Load = typename helpers::io::Load<DataType>;
  using Store = typename helpers::io::Store<DataType>;

  DepthwiseConv2D(Index n_filter_elems, Index n_b_items, Index n_k_items,
                  DepthwiseConv2DParams const& params,
                  ReadMem<T const, IsUSM> const& input,
                  ReadMem<T const, IsUSM> const& filter,
                  LocalAccessor<T> const& local,
                  WriteMem<T, IsUSM> const& output)
      : n_filter_elems_{n_filter_elems / VectorWidth},
        features_{params.channels * params.channel_multiplier},
        workgroup_batch_items_{n_b_items},
        workgroup_col_items_{n_k_items},
        p_{params},
        input_values_{input},
        output_errors_{filter},
        workspace_{local},
        filter_output_{output} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::nd_item<2> item) const {
    Index const local_idx = item.get_global_id(0);
    Index const fil_idx = item.get_global_id(1);

    DataType out_val{0};
    if (fil_idx < n_filter_elems_) {
      auto const input_data = input_values_.get_pointer();
      auto const error_data = output_errors_.get_pointer();

      auto const workgroup_idx =
          helpers::TensorIndexHelper<Index, false>::unflatten2d(
              local_idx, workgroup_col_items_, workgroup_col_items_);
      Index const k_idx = workgroup_idx.s1;
      Index const batch_idx = workgroup_idx.s0;

      auto const filter_tensor_idx =
          helpers::TensorIndexHelper<Index, false>::unflatten4d(
              fil_idx, p_.out_cols, p_.out_cols, p_.channels / VectorWidth,
              p_.channels / VectorWidth, p_.channel_multiplier,
              p_.channel_multiplier);
      Index const multiple = filter_tensor_idx.s3;
      Index const channel = filter_tensor_idx.s2 * VectorWidth;
      Index const col_idx = filter_tensor_idx.s1;
      Index const row_idx = filter_tensor_idx.s0;

      auto const col_window_struct =
          helpers::in_window_from_output(col_idx, 1, p_.pad_cols);
      Index const cstart =
          col_window_struct.window_start + (k_idx * p_.stride_cols);
      Index const firstc = col_window_struct.filter_start + k_idx;

      auto const row_window_struct =
          helpers::in_window_from_output(row_idx, 1, p_.pad_rows);
      Index const rstart = row_window_struct.window_start;
      Index const firstr = row_window_struct.filter_start;

      Index const input_initial_offset =
          batch_idx * p_.in_cols * p_.in_rows * p_.channels + channel;
      Index const error_initial_offset =
          channel * p_.channel_multiplier + multiple;

      Index input_batch_offset = input_initial_offset;
      Index error_batch_offset =
          error_initial_offset +
          batch_idx * p_.window_rows * p_.window_cols * features_;
      for (Index b = batch_idx; b < p_.batch; b += workgroup_batch_items_) {
        Index input_row_offset =
            input_batch_offset + rstart * p_.in_cols * p_.channels;
        Index error_row_offset =
            error_batch_offset + firstr * p_.window_cols * features_;

        for (Index row = rstart, i = firstr; i < p_.window_rows;
             ++i, row += p_.stride_rows) {
          if (row >= 0 && row < p_.in_rows) {
            Index input_col_offset = input_row_offset + cstart * p_.channels;
            Index error_col_offset = error_row_offset + firstc * features_;

            for (Index col = cstart, j = firstc; j < p_.window_cols;
                 j += workgroup_col_items_,
                       col += (workgroup_col_items_ * p_.stride_cols)) {
              if (col >= 0 && col < p_.in_cols) {
                DataType in_val = Load()(input_data, input_col_offset);
                DataType fil_val = Load()(error_data, error_col_offset);

                out_val = helpers::math::mad(in_val, fil_val, out_val);
              }

              input_col_offset +=
                  workgroup_col_items_ * p_.stride_cols * p_.channels;
              error_col_offset += workgroup_col_items_ * features_;
            }  // col loop
          }

          input_row_offset += p_.stride_rows * p_.in_cols * p_.channels;
          error_row_offset += p_.window_cols * features_;
        }  // row loop

        input_batch_offset +=
            workgroup_batch_items_ * p_.in_rows * p_.in_cols * p_.channels;
        error_batch_offset += workgroup_batch_items_ * p_.window_rows *
                              p_.window_cols * features_;
      }  // batch loop

    }  // if (fil_idx < n_filter_elems_)

    // The reduce has to be outside any conditional, to ensure that all threads
    // reach the barriers used in the reduction.
    out_val = helpers::reduce::workgroup_reduce<helpers::reduce::Sum, Index>(
        out_val, item,
        workspace_.template get_multi_ptr<sycl::access::decorated::legacy>());

    if (local_idx == 0 && fil_idx < n_filter_elems_) {
      auto output_data = filter_output_.get_pointer();
      Store()(output_data, fil_idx * VectorWidth, out_val);
    }
  }

 private:
  Index const n_filter_elems_;
  Index const features_;
  Index const workgroup_batch_items_;
  Index const workgroup_col_items_;
  DepthwiseConv2DParams const p_;
  ReadMem<T const, IsUSM> const input_values_;
  ReadMem<T const, IsUSM> const output_errors_;
  LocalAccessor<T> workspace_;
  WriteMem<T, IsUSM> filter_output_;
};

}  // namespace internal
}  // namespace depthwise_conv2d
}  // namespace sycldnn

#endif  // PORTDNN_SRC_DEPTHWISE_CONV2D_KERNELS_H_
