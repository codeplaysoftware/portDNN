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
#ifndef PORTDNN_SRC_CONV2D_IM2COL_KERNELS_H_
#define PORTDNN_SRC_CONV2D_IM2COL_KERNELS_H_

#include "portdnn/accessor_types.h"

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/params.h"

#include "portdnn/helpers/macros.h"

#include "src/helpers/tensor_index.h"
#include "src/helpers/vector_io.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace im2col {

template <typename T, typename Index, bool isUSM>
struct ExtractFilterTiles {
  using Load = helpers::io::Load<T>;
  using Store = helpers::io::Store<T>;

  ExtractFilterTiles(Conv2DParams const& params,
                     ReadMem<T const, isUSM> const& input,
                     WriteMem<T, isUSM> const& output)
      : n_items_{params.window_rows * params.window_cols * params.channels *
                 params.features},
        n_window_rows_{params.window_rows},
        n_window_cols_{params.window_cols},
        n_channels_{params.channels},
        n_features_{params.features},
        input_mem_{input},
        output_mem_{output} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<1> item) const {
    Index index = item.get_id(0);

    if (index < n_items_) {
      auto input_data = input_mem_.get_pointer();
      auto output_data = output_mem_.get_pointer();

      T in_val = Load()(input_data, index);

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
      Store()(output_data, out_idx, in_val);
    }
  }

 private:
  Index const n_items_;
  Index const n_window_rows_;
  Index const n_window_cols_;
  Index const n_channels_;
  Index const n_features_;
  ReadMem<T const, isUSM> input_mem_;
  WriteMem<T, isUSM> output_mem_;
};

}  // namespace im2col
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_SRC_CONV2D_IM2COL_KERNELS_H_
