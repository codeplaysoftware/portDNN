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

#ifndef PORTDNN_SRC_POOLING_KERNELS_H_
#define PORTDNN_SRC_POOLING_KERNELS_H_

#include <CL/sycl.hpp>

#include "src/helpers/fast_div.h"
#include "src/helpers/tensor_index.h"
#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"
#include "src/pooling/operators_impl.h"

#include "portdnn/accessor_types.h"
#include "portdnn/format_type.h"

#include "portdnn/helpers/minmax.h"

#include "portdnn/pooling/params.h"

namespace sycldnn {
namespace pooling {

template <typename T, typename Index, template <typename> class Op,
          typename Direction, int VectorWidth, bool UseFastDiv, typename Layout,
          bool IsUSM>
class PoolingOp;

template <typename T, typename Index, template <typename> class Op,
          int VectorWidth, bool UseFastDiv, bool IsUSM>
class PoolingOp<T, Index, Op, Forward, VectorWidth, UseFastDiv, layout::NHWC,
                IsUSM> {
  using IndexDivType = typename fast_div::IndexDiv<Index, UseFastDiv>::type;
  using DataT = typename helpers::VectorType<T, VectorWidth>::type;
  using Load = helpers::io::Load<DataT>;
  using Store = helpers::io::Store<DataT>;

  ReadMem<T const, IsUSM> in_data_;
  WriteMem<T, IsUSM> out_data_;
  const Index n_items_;
  PoolingParams params_;
  const IndexDivType div_out_rows_;
  const IndexDivType div_out_cols_;
  const IndexDivType div_channels_;

 public:
  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) const {
    Index index = item.get_id(0);

    if (index < n_items_) {
      Op<DataT> op;
      const auto in_data = in_data_.get_pointer();
      const auto out_data = out_data_.get_pointer();

      const auto tensor_id =
          helpers::TensorIndexHelper<Index, UseFastDiv>::unflatten4d(
              index, div_out_rows_, params_.out_rows, div_out_cols_,
              params_.out_cols, div_channels_, params_.channels / VectorWidth);
      const auto feature = tensor_id.s3 * VectorWidth;
      const auto col = tensor_id.s2;
      const auto row = tensor_id.s1;
      const auto batch = tensor_id.s0;

      auto row_start = row * params_.stride_rows - params_.pad_rows;
      const auto row_end =
          helpers::min(row_start + params_.window_rows, params_.in_rows);
      row_start = helpers::max(row_start, 0);

      auto col_start = col * params_.stride_cols - params_.pad_cols;
      const auto col_end =
          helpers::min(col_start + params_.window_cols, params_.in_cols);
      col_start = helpers::max(col_start, 0);

      const auto input_offset =
          batch * params_.in_cols * params_.in_rows * params_.channels;
      const auto offset_pointer = in_data + input_offset;
      for (Index r = row_start; r < row_end; r++) {
        for (Index c = col_start; c < col_end; c++) {
          Index loc = (r * params_.in_cols + c) * params_.channels + feature;
          op.accumulate(Load()(offset_pointer, loc));
        }
      }
      Store()(out_data, index * VectorWidth, op.value());
    }
  }

  PoolingOp(ReadMem<T const, IsUSM> in_data, WriteMem<T, IsUSM> out_data,
            PoolingParams const& pp)
      : in_data_(std::move(in_data)),
        out_data_(std::move(out_data)),
        n_items_(pp.batch * pp.out_rows * pp.out_cols * pp.channels /
                 VectorWidth),
        params_(pp),
        div_out_rows_{pp.out_rows},
        div_out_cols_{pp.out_cols},
        div_channels_{pp.channels / VectorWidth} {}
};

/**
 * Max pooling gradient kernel.
 *
 * Expects to be run with one thread per output value in the backprop kernel.
 */
template <typename T, typename Index, template <typename> class MaxOp,
          int VectorWidth, bool UseFastDiv, bool IsUSM>
class PoolingOp<T, Index, MaxOp, Backpropagate, VectorWidth, UseFastDiv,
                layout::NHWC, IsUSM> {
  using DataType = typename helpers::VectorType<T, 1>::type;
  using LoadData = helpers::io::Load<DataType>;
  using StoreData = helpers::io::Store<DataType>;
  using IndexDivType = typename fast_div::IndexDiv<Index, UseFastDiv>::type;

 public:
  PoolingOp(ReadMem<T const, IsUSM> const& in_data,
            ReadMem<T const, IsUSM> const& out_data,
            ReadMem<T const, IsUSM> const& in_backprop,
            WriteMem<T, IsUSM> const& out_backprop, PoolingParams const& pp)
      : in_data_{in_data},
        out_data_{out_data},
        in_backprop_{in_backprop},
        out_backprop_{out_backprop},
        n_items_{pp.batch * pp.in_rows * pp.in_cols * pp.channels},
        params_{pp},
        div_in_rows_{pp.in_rows},
        div_in_cols_{pp.in_cols},
        div_channels_{pp.channels} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<1> item) const {
    Index index = item.get_id(0);

    if (index < n_items_) {
      auto in_data = in_data_.get_pointer();
      auto out_data = out_data_.get_pointer();
      auto in_backprop = in_backprop_.get_pointer();
      auto out_backprop = out_backprop_.get_pointer();

      const auto tensor_id =
          helpers::TensorIndexHelper<Index, UseFastDiv>::unflatten4d(
              index, div_in_rows_, params_.in_rows, div_in_cols_,
              params_.in_cols, div_channels_, params_.channels);
      auto const channel = tensor_id.s3;
      auto const col_idx = tensor_id.s2 + params_.pad_cols;
      auto const row_idx = tensor_id.s1 + params_.pad_rows;
      auto const batch = tensor_id.s0;
      DataType gradient{0};
      auto const input_value = LoadData()(in_data, index);

      auto const col_input = get_input_window(
          col_idx, params_.out_cols, params_.window_cols, params_.stride_cols);
      auto const row_input = get_input_window(
          row_idx, params_.out_rows, params_.window_rows, params_.stride_rows);

      Index const index_no_n =
          index - batch * params_.in_cols * params_.in_rows * params_.channels -
          channel;

      auto const input_data_n =
          in_data +
          batch * params_.in_cols * params_.in_rows * params_.channels +
          channel;
      auto const output_data_n =
          out_data +
          batch * params_.out_cols * params_.out_rows * params_.channels +
          channel;
      auto const input_backprop_n =
          in_backprop +
          batch * params_.out_cols * params_.out_rows * params_.channels +
          channel;

      for (Index poolr = row_input.begin; poolr < row_input.end; ++poolr) {
        auto const row_output =
            get_output_window(poolr, params_.in_rows, params_.window_rows,
                              params_.stride_rows, params_.pad_rows);
        for (Index poolc = col_input.begin; poolc < col_input.end; ++poolc) {
          auto const col_output =
              get_output_window(poolc, params_.in_cols, params_.window_cols,
                                params_.stride_cols, params_.pad_cols);

          Index const output_data_idx =
              (poolr * params_.out_cols + poolc) * params_.channels;
          auto const output_value = LoadData()(output_data_n, output_data_idx);

          bool is_max = EqualCheck<MaxOp>::are_equal(input_value, output_value);
          bool should_continue = is_max;

          // Even if this thread's output value is the maximum, we cannot say
          // for sure that the input gradient should be assigned to this
          // thread's output gradient. It could be the case that another input
          // in the output value's pool is also the maximum, and then we need to
          // assign the gradient to the first maximum value.
          //
          // To ensure that we assign the gradient to the correct output, loop
          // through the pool values which appear before this thread's value and
          // check that none of those values are a maximum.
          //
          // This is unlikely to occur in real life, as the chances of two
          // random floats in a max pool coinciding is rare, however Tensorflow
          // contains tests which explicitly set input values to be the same.
          // Perhaps this check should be optional to allow a user to choose
          // performance in the general case over correctness in the rare case
          // that this is needed.
          // TODO(jwlawson): Add option to disable max pool correctness check.
          for (Index win_r = row_output.begin;
               win_r < row_output.end && should_continue; ++win_r) {
            for (Index win_c = col_output.begin;
                 win_c < col_output.end && should_continue; ++win_c) {
              Index const input_data_idx =
                  (win_r * params_.in_cols + win_c) * params_.channels;

              if (input_data_idx == index_no_n) {
                // Only check up to the input index
                should_continue = false;
              } else {
                DataType next_val = LoadData()(input_data_n, input_data_idx);
                if (EqualCheck<MaxOp>::are_equal(next_val, output_value)) {
                  // Found another maximum value before this thread's value
                  should_continue = false;
                  is_max = false;
                }
              }
            }
          }
          if (is_max) {
            gradient += LoadData()(input_backprop_n, output_data_idx);
          }
        }
      }
      StoreData()(out_backprop, index, gradient);
    }
  }

 private:
  /** Struct defining a window in one dimension of a tensor. */
  struct Window {
    /** First index into the window. */
    Index begin;
    /** One past the last index into the window. */
    Index end;
  };

  /** Get the input window corresponding to the given index.  */
  Window SNN_ALWAYS_INLINE get_input_window(Index idx, Index max_idx,
                                            Index window_size,
                                            Index stride) const {
    Index const begin =
        (idx < window_size) ? 0 : (idx - window_size) / stride + 1;
    Index const end = helpers::min(idx / stride + 1, max_idx);
    return Window{begin, end};
  }

  /** Get the output window corresponding to the given index.  */
  Window SNN_ALWAYS_INLINE get_output_window(Index idx, Index max_idx,
                                             Index window_size, Index stride,
                                             Index pad) const {
    Index begin = idx * stride - pad;
    Index end = helpers::min(begin + window_size, max_idx);
    begin = helpers::max(begin, 0);
    return Window{begin, end};
  }

  ReadMem<T const, IsUSM> in_data_;
  ReadMem<T const, IsUSM> out_data_;
  ReadMem<T const, IsUSM> in_backprop_;
  WriteMem<T, IsUSM> out_backprop_;
  Index n_items_;
  PoolingParams params_;
  const IndexDivType div_in_rows_;
  const IndexDivType div_in_cols_;
  const IndexDivType div_channels_;
};

/**
 * Average pooling gradient kernel.
 *
 * Expects to be run with one thread per output value in the backprop kernel.
 */
template <typename T, typename Index, int VectorWidth, bool UseFastDiv,
          bool IsUSM>
class PoolingOp<T, Index, Average, Backpropagate, VectorWidth, UseFastDiv,
                layout::NHWC, IsUSM> {
  using DataType = typename helpers::VectorType<T, VectorWidth>::type;
  using LoadData = helpers::io::Load<DataType>;
  using StoreData = helpers::io::Store<DataType>;
  using IndexDivType = typename fast_div::IndexDiv<Index, UseFastDiv>::type;

 public:
  PoolingOp(ReadMem<T const, IsUSM> const& in_data,
            WriteMem<T, IsUSM> const& out_data, PoolingParams const& pp)
      : in_backprop_{in_data},
        out_backprop_{out_data},
        n_items_{pp.batch * pp.in_rows * pp.in_cols * pp.channels /
                 VectorWidth},
        params_{pp},
        div_in_rows_{pp.in_rows},
        div_in_cols_{pp.in_cols},
        div_channels_{pp.channels / VectorWidth} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<1> item) const {
    Index index = item.get_id(0);

    if (index < n_items_) {
      auto input_backprop = in_backprop_.get_pointer();
      auto output_backprop = out_backprop_.get_pointer();

      const auto tensor_id =
          helpers::TensorIndexHelper<Index, UseFastDiv>::unflatten4d(
              index, div_in_rows_, params_.in_rows, div_in_cols_,
              params_.in_cols, div_channels_, params_.channels / VectorWidth);
      auto const channel = tensor_id.s3 * VectorWidth;
      auto const col_idx = tensor_id.s2 + params_.pad_cols;
      auto const row_idx = tensor_id.s1 + params_.pad_rows;
      auto const batch = tensor_id.s0;

      auto const col_input = get_input_window(
          col_idx, params_.out_cols, params_.window_cols, params_.stride_cols);
      auto const row_input = get_input_window(
          row_idx, params_.out_rows, params_.window_rows, params_.stride_rows);

      DataType gradient{0};
      auto input_backprop_n =
          input_backprop +
          batch * params_.out_cols * params_.out_rows * params_.channels +
          channel;

      // For each element in the input window compute the size of the
      // corresponding average pool window. The pool window may include some
      // padding, which are discounted from the pooling, so the window size may
      // not correspond exactly to the parameter's window sizes.
      //
      // Each input gradient needs to be spread out across all the indicies
      // which contributed to that average pool output value, and so we divide
      // the input gradients by the window size before accumulating in the
      // output.
      for (Index poolr = row_input.begin; poolr < row_input.end; ++poolr) {
        Index const row_window_size =
            get_actual_window_size(poolr, params_.in_rows, params_.window_rows,
                                   params_.stride_rows, params_.pad_rows);

        for (Index poolc = col_input.begin; poolc < col_input.end; ++poolc) {
          Index const col_window_size = get_actual_window_size(
              poolc, params_.in_cols, params_.window_cols, params_.stride_cols,
              params_.pad_cols);

          Index const idx =
              (poolr * params_.out_cols + poolc) * params_.channels;
          Index const window_size = row_window_size * col_window_size;
          gradient +=
              LoadData()(input_backprop_n, idx) / static_cast<T>(window_size);
        }
      }
      StoreData()(output_backprop, index * VectorWidth, gradient);
    }
  }

 private:
  /**
   * Compute the actual size of a window for the given index.
   *
   * The window may fall off the start or end of the tensor, so the size may be
   * smaller than just the value of `window_size`.
   */
  Index SNN_ALWAYS_INLINE get_actual_window_size(Index idx, Index max_idx,
                                                 Index window_size,
                                                 Index stride,
                                                 Index pad) const {
    Index start = idx * stride - pad;
    Index const end = helpers::min(start + window_size, max_idx);
    start = helpers::max(start, 0);
    Index const size = end - start;
    return size;
  }

  /** Struct defining a window in one dimension of the input tensor. */
  struct InputWindow {
    /** First index into the window. */
    Index begin;
    /** One past the last index into the window. */
    Index end;
  };

  /**
   * Get the input window corresponding to the given index.
   */
  InputWindow SNN_ALWAYS_INLINE get_input_window(Index idx, Index max_idx,
                                                 Index window_size,
                                                 Index stride) const {
    Index const begin =
        (idx < window_size) ? 0 : (idx - window_size) / stride + 1;
    Index const end = helpers::min(idx / stride + 1, max_idx);
    return InputWindow{begin, end};
  }

  ReadMem<T const, IsUSM> in_backprop_;
  WriteMem<T, IsUSM> out_backprop_;
  Index n_items_;
  PoolingParams params_;
  const IndexDivType div_in_rows_;
  const IndexDivType div_in_cols_;
  const IndexDivType div_channels_;
};

template <typename T, typename Index, template <typename> class Op,
          bool UseFastDiv, bool IsUSM>
class PoolingOp<T, Index, Op, Forward, /*VectorWidth=*/1, UseFastDiv,
                layout::NCHW, IsUSM> {
  using IndexDivType = typename fast_div::IndexDiv<Index, UseFastDiv>::type;
  using Load = helpers::io::Load<T>;
  using Store = helpers::io::Store<T>;

  ReadMem<T const, IsUSM> in_data_;
  WriteMem<T, IsUSM> out_data_;
  const Index n_items_;
  PoolingParams params_;
  const IndexDivType div_out_rows_;
  const IndexDivType div_out_cols_;
  const IndexDivType div_channels_;

 public:
  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) const {
    Index index = item.get_id(0);

    if (index < n_items_) {
      Op<T> op;
      const auto in_data = in_data_.get_pointer();
      const auto out_data = out_data_.get_pointer();

      const auto tensor_id =
          helpers::TensorIndexHelper<Index, UseFastDiv>::unflatten4d(
              index, div_channels_, params_.channels, div_out_rows_,
              params_.out_rows, div_out_cols_, params_.out_cols);
      const auto col = tensor_id.s3;
      const auto row = tensor_id.s2;
      const auto feature = tensor_id.s1;
      const auto batch = tensor_id.s0;

      auto row_start = row * params_.stride_rows - params_.pad_rows;
      const auto row_end =
          helpers::min(row_start + params_.window_rows, params_.in_rows);
      row_start = helpers::max(row_start, 0);

      auto col_start = col * params_.stride_cols - params_.pad_cols;
      const auto col_end =
          helpers::min(col_start + params_.window_cols, params_.in_cols);
      col_start = helpers::max(col_start, 0);

      const auto input_offset =
          batch * params_.in_cols * params_.in_rows * params_.channels;
      const auto offset_pointer = in_data + input_offset;
      for (Index r = row_start; r < row_end; r++) {
        for (Index c = col_start; c < col_end; c++) {
          Index loc = (feature * params_.in_rows + r) * params_.in_cols + c;
          op.accumulate(Load()(offset_pointer, loc));
        }
      }
      Store()(out_data, index, op.value());
    }
  }

  PoolingOp(ReadMem<T const, IsUSM> in_data, WriteMem<T, IsUSM> out_data,
            PoolingParams const& pp)
      : in_data_(std::move(in_data)),
        out_data_(std::move(out_data)),
        n_items_(pp.batch * pp.out_rows * pp.out_cols * pp.channels),
        params_(pp),
        div_out_rows_{pp.out_rows},
        div_out_cols_{pp.out_cols},
        div_channels_{pp.channels} {}
};

}  // namespace pooling
}  // namespace sycldnn

#endif  // PORTDNN_SRC_POOLING_KERNELS_H_
