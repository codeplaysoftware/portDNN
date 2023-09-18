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
#ifndef PORTDNN_SRC_TRANSPOSE_KERNELS_H_
#define PORTDNN_SRC_TRANSPOSE_KERNELS_H_

#include "portdnn/accessor_types.h"
#include "portdnn/status.h"

#include "src/helpers/vector_io.h"

#include <algorithm>
#include <numeric>

namespace sycldnn {
namespace transpose {
namespace internal {

template <typename T, typename Index, int ND, bool IsUSM>
struct TransposeKernel {
  using LoadData = helpers::io::Load<T>;
  using StoreData = helpers::io::Store<T>;

  TransposeKernel(ReadMem<T const, IsUSM> const& input,
                  WriteMem<T, IsUSM> const& output,
                  std::vector<int> const& dimensions,
                  std::vector<int> const& permutation)
      : input_{input},
        output_{output},
        tensor_size_{std::accumulate(begin(dimensions), end(dimensions),
                                     static_cast<Index>(1),
                                     [](Index a, int b) { return a * b; })} {
    std::copy_n(begin(dimensions), ND, begin(in_dims_));
    std::copy_n(begin(permutation), ND, begin(permutation_));
    for (int i = 0; i < ND; ++i) {
      out_dims_[i] = dimensions[permutation[i]];
    }
  };

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<1> item) const {
    Index flat_in_id = item.get_id(0);
    if (flat_in_id < tensor_size_) {
      auto in_ptr = input_.get_pointer();
      auto out_ptr = output_.get_pointer();

      auto in_val = LoadData()(in_ptr, flat_in_id);

      std::array<int, ND> in_id;
      for (int i = ND - 1; i >= 0; --i) {
        in_id[i] = flat_in_id % in_dims_[i];
        flat_in_id /= in_dims_[i];
      }

      std::array<int, ND> out_id;
      for (int i = 0; i < ND; ++i) {
        out_id[i] = in_id[permutation_[i]];
      }

      Index flat_out_id = out_id[0];
      for (int i = 1; i < ND; ++i) {
        flat_out_id *= out_dims_[i];
        flat_out_id += out_id[i];
      }

      StoreData()(out_ptr, flat_out_id, in_val);
    }
  }

 private:
  ReadMem<T const, IsUSM> input_;
  WriteMem<T, IsUSM> output_;
  Index tensor_size_;
  std::array<int, ND> in_dims_;
  std::array<int, ND> out_dims_;
  std::array<int, ND> permutation_;
};

}  // namespace internal
}  // namespace transpose
}  // namespace sycldnn

#endif  // PORTDNN_SRC_TRANSPOSE_KERNELS_H_
