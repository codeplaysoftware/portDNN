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
#ifndef PORTDNN_SRC_GATHER_KERNELS_H_
#define PORTDNN_SRC_GATHER_KERNELS_H_

#include <CL/sycl.hpp>

#include "src/helpers/tensor_index.h"
#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"

#include "portdnn/accessor_types.h"

#include "portdnn/gather/params.h"
#include "portdnn/gather/sizes.h"

namespace sycldnn {
namespace gather {

template <typename T, typename Index, bool IsUSM>
class GatherOp {
  using LoadData = helpers::io::Load<T>;
  using StoreData = helpers::io::Store<T>;

  ReadMem<T const, IsUSM> in_data_;
  ReadMem<Index const, IsUSM> indices_data_;
  WriteMem<T, IsUSM> out_data_;

  const Index block_size_;
  const Index max_indices_;
  const Index n_indices_;
  const Index n_items_;

 public:
  GatherOp(ReadMem<T const, IsUSM> const& input,
           ReadMem<Index const, IsUSM> const& indices,
           WriteMem<T, IsUSM> const& output, Index block_size,
           Index max_indices, Index n_indices, Index n_items)
      : in_data_{input},
        indices_data_{indices},
        out_data_{output},
        block_size_{block_size},
        max_indices_{max_indices},
        n_indices_{n_indices},
        n_items_{n_items} {}

  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) const {
    Index idx = item.get_id(0);

    Index out_block_id = idx / (block_size_ * n_indices_);

    Index relative_idx = idx - out_block_id * block_size_ * n_indices_;

    Index block_relative_idx = relative_idx % block_size_;

    Index out_index = idx / block_size_ - out_block_id * n_indices_;

    auto in_ptr = in_data_.get_pointer();
    Index const* indices_ptr = indices_data_.get_pointer();
    auto out_ptr = out_data_.get_pointer();

    auto index_value = indices_ptr[out_index];
    index_value += (index_value < 0) ? max_indices_ : 0;

    if (idx < n_items_ && index_value >= 0 && index_value < max_indices_) {
      auto in_id = index_value * block_size_ +
                   out_block_id * block_size_ * max_indices_ +
                   block_relative_idx;

      auto in_val = LoadData()(in_ptr, in_id);

      StoreData()(out_ptr, idx, in_val);
    } else if (idx < n_items_) {
      StoreData()(out_ptr, idx, (T)0);
    }
  }
};

}  // namespace gather
}  // namespace sycldnn

#endif  // PORTDNN_SRC_GATHER_KERNELS_H_
