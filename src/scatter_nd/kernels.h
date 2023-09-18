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

#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"

#include "portdnn/accessor_types.h"

#include "helpers.h"
#include "portdnn/scatter_nd/operators.h"
#include "portdnn/scatter_nd/sizes.h"

#ifndef PORTDNN_SRC_SCATTER_ND_KERNELS_H_
#define PORTDNN_SRC_SCATTER_ND_KERNELS_H_
namespace sycldnn {
namespace scatter_nd {

template <typename MultiPtr, typename DataType, typename IndexType>
void Assign::apply(MultiPtr& ptr, IndexType offset, DataType val) {
  helpers::io::Store<DataType>()(ptr, offset, val);
}

template <typename MultiPtr, typename DataType, typename IndexType>
void Add::apply(MultiPtr& ptr, IndexType offset, DataType val) {
  *(ptr + offset) += val;
}

template <typename MultiPtr, typename DataType, typename IndexType>
void Sub::apply(MultiPtr& ptr, IndexType offset, DataType val) {
  *(ptr + offset) -= val;
}

template <typename MultiPtr, typename DataType, typename IndexType>
void Mul::apply(MultiPtr& ptr, IndexType offset, DataType val) {
  *(ptr + offset) *= val;
}

template <typename MultiPtr, typename DataType, typename IndexType>
void Div::apply(MultiPtr& ptr, IndexType offset, DataType val) {
  *(ptr + offset) /= val;
}

template <typename DType, typename IType, typename UpdateOp, int IndexDepth,
          int VectorWidth, bool IsUSM>
class ScatterNDOp {
  using DataType = typename helpers::VectorType<DType, VectorWidth>::type;
  using LoadData = helpers::io::Load<DataType>;
  ReadMem<IType const, IsUSM> ind_data_;
  ReadMem<DType const, IsUSM> upd_data_;
  WriteMem<DType, IsUSM> out_data_;
  IndexHelper<IndexDepth> index_helper_;
  size_t slice_size_;
  size_t n_updates_;

 public:
  ScatterNDOp(ReadMem<IType const, IsUSM> ind_data,
              ReadMem<DType const, IsUSM> upd_data,
              WriteMem<DType, IsUSM> out_data, ScatterNDSizes const& ss)
      : ind_data_(ind_data),
        upd_data_(upd_data),
        out_data_(out_data),
        index_helper_(ss.dim_0, ss.dim_1, ss.dim_2, ss.dim_3),
        slice_size_(ss.slice_size / VectorWidth),
        n_updates_(ss.num_updates) {}
  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<2> item) const {
    const IType update_row = item.get_id(0);
    const IType update_col = item.get_id(1);

    if (update_row < static_cast<IType>(n_updates_) &&
        update_col < static_cast<IType>(slice_size_)) {
      // Get pointer to index data
      auto ind_ptr = ind_data_.get_pointer();
      // Calculate the offset for the start of the slice in the output tensor
      IType output_offset =
          index_helper_(ind_ptr, update_row) + update_col * VectorWidth;

      // index_helper_ returns -1 if the index is out of bounds. Following
      // tensorflow we just don't apply that update
      if (output_offset == -1) {
        return;
      }

      // Get pointers for update and output tensors
      auto update_ptr = upd_data_.get_pointer();
      auto output_ptr = out_data_.get_pointer();

      // Calculate the offset in the update data that points to the chunk of
      // data to be copied.
      IType update_offset =
          (update_row * slice_size_ + update_col) * VectorWidth;
      // Load the chunk of data. If IndexDepth==rank(input) (i.e. performing an
      // elementwise update) or UpdateOp/=Assign then the chunk is just a single
      // element, else it is a chunk the size of VectorWidth.
      DataType update_val = LoadData()(update_ptr, update_offset);

      UpdateOp::apply(output_ptr, output_offset, update_val);
    }
  }
};
}  // namespace scatter_nd
}  // namespace sycldnn
#endif  // PORTDNN_SRC_SCATTER_ND_KERNELS_H_
