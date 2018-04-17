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
#ifndef SYCLDNN_SRC_CONV2D_IM2COL_KERNELS_H_
#define SYCLDNN_SRC_CONV2D_IM2COL_KERNELS_H_

#include "sycldnn/accessor_types.h"

#include "sycldnn/conv2d/conv_type.h"
#include "sycldnn/conv2d/params.h"

#include "sycldnn/helpers/macros.h"

#include "src/helpers/tensor_index.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace im2col {

template <typename T, typename Index>
struct ExtractFilterTiles {
  ExtractFilterTiles(Index in_offset, Index out_offset,
                     Conv2DParams const& params,
                     ReadAccessor<T const> const& input,
                     WriteAccessor<T> const& output)
      : n_items_{params.window_rows * params.window_cols * params.channels *
                 params.features},
        in_offset_{in_offset},
        out_offset_{out_offset},
        n_window_rows_{params.window_rows},
        n_window_cols_{params.window_cols},
        n_channels_{params.channels},
        n_features_{params.features},
        input_accessor_{input},
        output_accessor_{output} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<1> item) {
    Index index = item.get_id(0);
    if (index < n_items_) {
      T const* input_data = input_accessor_.get_pointer().get() + in_offset_;
      T* output_data = output_accessor_.get_pointer().get() + out_offset_;

      T in_val = input_data[index];

      auto const tensor_idx =
          helpers::TensorIndexHelper<Index, false>::unflatten4d(
              index, n_window_cols_, n_window_cols_, n_channels_, n_channels_,
              n_features_, n_features_);
      Index const feature = tensor_idx.s3;
      Index const channel = tensor_idx.s2;
      Index const col = tensor_idx.s1;
      Index const row = tensor_idx.s0;

      Index const out_row = n_window_rows_ - 1 - row;
      Index const out_col = n_window_cols_ - 1 - col;
      Index const out_idx =
          ((out_row * n_window_cols_ + out_col) * n_features_ + feature) *
              n_channels_ +
          channel;
      output_data[out_idx] = in_val;
    }
  }

 private:
  Index const n_items_;
  Index const in_offset_;
  Index const out_offset_;
  Index const n_window_rows_;
  Index const n_window_cols_;
  Index const n_channels_;
  Index const n_features_;
  ReadAccessor<T const> input_accessor_;
  WriteAccessor<T> output_accessor_;
};

}  // namespace im2col
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_CONV2D_IM2COL_KERNELS_H_
