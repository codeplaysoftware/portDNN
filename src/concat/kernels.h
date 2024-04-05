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

#ifndef SYCLDNN_SRC_CONCAT_KERNELS_H_
#define SYCLDNN_SRC_CONCAT_KERNELS_H_

#include <CL/sycl.hpp>

#include "sycldnn/accessor_types.h"
#include "sycldnn/helpers/macros.h"

#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"

namespace sycldnn {
namespace concat {

template <typename T, int VectorWidth, int TArrayMaxSize, bool IsUSM>
class ConcatBase {
 protected:
  std::array<ReadMem<T const, IsUSM>, TArrayMaxSize> input_;

  WriteMem<T, IsUSM> output_;

  const int chunk_offset_;

  // Each tensor's data occupies a block of 3 values:
  //   {chunk_size, input_offset, flat_size}
  ReadMem<int const, IsUSM> concat_meta_data_;

 public:
  ConcatBase(std::array<ReadMem<T const, IsUSM>, TArrayMaxSize>& input,
             WriteMem<T, IsUSM>& output, const int chunk_offset,
             ReadMem<int const, IsUSM>& concat_meta_data)
      : input_{input},
        output_{output},
        chunk_offset_{chunk_offset},
        concat_meta_data_{concat_meta_data} {}
};

template <typename T, int VectorWidth, int TArrayMaxSize, bool IsUSM>
class ConcatOp : public ConcatBase<T, VectorWidth, TArrayMaxSize, IsUSM> {
  using DataType = typename helpers::VectorType<T, VectorWidth>::type;
  using LoadData = helpers::io::Load<DataType>;
  using StoreData = helpers::io::Store<DataType>;

 public:
  ConcatOp(std::array<ReadMem<T const, IsUSM>, TArrayMaxSize>& input,
           WriteMem<T, IsUSM>& output, const int chunk_offset,
           ReadMem<int const, IsUSM> concat_meta_data)
      : ConcatBase<T, VectorWidth, TArrayMaxSize, IsUSM>(
            input, output, chunk_offset, concat_meta_data) {}

  /**
   * Along the concatenation axis, we split the current tensor into chunks, then
   * for each input interchangeably add each chunk into the output.
   */
  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<2> item) const {
    const int tensor_idx = item.get_id(0);
    const int vec_idx = item.get_id(1) * VectorWidth;

    // Vectorised load of meta-data block: {chunk_size, input_offset, flat_size}
    auto meta_data = helpers::io::Load<helpers::VectorType<int, 3>::type>()(
        this->concat_meta_data_.get_pointer(), tensor_idx * 3);

    if (vec_idx < meta_data[2]) {
      DataType in_vec =
          LoadData()(this->input_[tensor_idx].get_pointer(), vec_idx);

      bool isContiguous = true;
      std::array<int, VectorWidth> output_indexes;
      SNN_PRAGMA_UNROLL
      for (int i = 0; i < VectorWidth; i++) {
        const int index_within_tensor = vec_idx + i;
        const int chunk_index = index_within_tensor / meta_data[0];
        const int inner_index =
            index_within_tensor - (chunk_index * meta_data[0]);
        output_indexes[i] =
            (this->chunk_offset_ * chunk_index) + meta_data[1] + inner_index;

        /**
         * If all output offsets for vector are contiguous, can perform
         * vectorised store.
         */
        if (i != 0 && output_indexes[i] - output_indexes[i - 1] != 1) {
          isContiguous = false;
        }
      }

      if (isContiguous) {
        StoreData()(this->output_.get_pointer(), output_indexes[0], in_vec);
      } else {
        SNN_PRAGMA_UNROLL
        for (int i = 0; i < VectorWidth; i++) {
          this->output_.get_pointer()[output_indexes[i]] = in_vec[i];
        }
      }
    }
  }
};

/**
 * Template specialisation for case where VectorWidth is 1.
 * Removes all of the logic for determining whether a vectorised store can be
 * performed.
 */
template <typename T, int TArrayMaxSize, bool IsUSM>
class ConcatOp<T, 1, TArrayMaxSize, IsUSM>
    : public ConcatBase<T, 1, TArrayMaxSize, IsUSM> {
 public:
  ConcatOp(std::array<ReadMem<T const, IsUSM>, TArrayMaxSize>& input,
           WriteMem<T, IsUSM>& output, const int chunk_offset,
           ReadMem<int const, IsUSM> concat_meta_data)
      : ConcatBase<T, 1, TArrayMaxSize, IsUSM>(input, output, chunk_offset,
                                               concat_meta_data) {}

  int SNN_ALWAYS_INLINE compute_output_offset(const int index_within_tensor,
                                              const int tensor_idx) const {
    auto meta_data = helpers::io::Load<helpers::VectorType<int, 3>::type>()(
        this->concat_meta_data_.get_pointer(), tensor_idx * 3);
    const int chunk_index = index_within_tensor / meta_data[0];
    const int inner_index = index_within_tensor - (chunk_index * meta_data[0]);
    const int output_offset =
        (this->chunk_offset_ * chunk_index) + meta_data[1] + inner_index;
    return output_offset;
  }

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<2> item) const {
    const int tensor_idx = item.get_id(0);
    const int index_within_tensor = item.get_id(1);

    auto meta_data = helpers::io::Load<helpers::VectorType<int, 3>::type>()(
        this->concat_meta_data_.get_pointer(), tensor_idx * 3);

    if (index_within_tensor < meta_data[2]) {
      const int output_offset =
          compute_output_offset(index_within_tensor, tensor_idx);

      this->output_.get_pointer()[output_offset] =
          this->input_[tensor_idx].get_pointer()[index_within_tensor];
    }
  }
};

}  // namespace concat
}  // namespace sycldnn

#endif
