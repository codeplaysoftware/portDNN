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

#ifndef SYCLDNN_SRC_POOLING_KERNELS_H_
#define SYCLDNN_SRC_POOLING_KERNELS_H_

#include <CL/sycl.hpp>

#include "src/helpers/tensor_index.h"

#include "sycldnn/accessor_types.h"
#include "sycldnn/helpers/minmax.h"
#include "sycldnn/pooling/operators.h"
#include "sycldnn/pooling/params.h"

namespace sycldnn {
namespace pooling {

template <typename T>
struct Max {
  T max;
  Max() : max(std::numeric_limits<T>::lowest()) {}
  void accumulate(T val) { val > max ? max = val : T(0); }
  T value() { return max; }
};

/** Template that will average a sequence of accumulated values. */
template <typename T>
struct Average {
  /** The number of values accumulated. */
  int tally;
  /** The sum of the accumulated values. */
  T sum;

  Average() : tally(0), sum(0) {}

  /** Increases the running total of the struct's accumulator.
   * \param val The next value to be added to the accumulator. */
  void accumulate(T val) {
    tally++;
    sum += val;
  }

  /** Observes the average, by dividing the sum  by the number of tallies.
   * \return The average of all accumulated values. */
  T value() { return sum / T(tally); }
};

template <typename T, typename Index, template <typename U> class Op,
          typename Direction>
class PoolingOp;

template <typename T, typename Index, template <typename U> class Op>
class PoolingOp<T, Index, Op, Forward> {
  ReadAccessor<T const> in_data_;
  WriteAccessor<T> out_data_;
  PoolingParams params_;

 public:
  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index index = item.get_id(0);
    if (index < params_.batch * params_.out_rows * params_.out_cols *
                    params_.channels) {
      Op<T> op;
      const auto tensor_id =
          helpers::TensorIndexHelper<Index, false>::unflatten4d(
              index, params_.out_rows, params_.out_rows, params_.out_cols,
              params_.out_cols, params_.channels, params_.channels);
      const auto feature = tensor_id.s3;
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

      const auto input_data_offset =
          in_data_.get_pointer() +
          batch * params_.in_cols * params_.in_rows * params_.channels;
      for (Index r = row_start; r < row_end; r++) {
        for (Index c = col_start; c < col_end; c++) {
          Index loc = (r * params_.in_cols + c) * params_.channels + feature;
          op.accumulate(input_data_offset.get()[loc]);
        }
      }
      out_data_[index] = op.value();
    }
  }

  PoolingOp(ReadAccessor<T const> in_data, WriteAccessor<T> out_data,
            const PoolingParams& pp)
      : in_data_(in_data), out_data_(out_data), params_(pp) {}
};

}  // namespace pooling
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_POOLING_KERNELS_H_
