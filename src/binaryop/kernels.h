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

#ifndef SYCLDNN_SRC_BINARYOP_KERNELS_H_
#define SYCLDNN_SRC_BINARYOP_KERNELS_H_

#include <CL/sycl.hpp>

#include "src/helpers/tensor_index.h"
#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"

#include "sycldnn/accessor_types.h"

#include "sycldnn/binaryop/operators.h"

namespace sycldnn {
namespace binaryop {

struct Add {
  template <typename T>
  inline T apply(T val1, T val2) {
    return val1 + val2;
  }
};

struct Sub {
  template <typename T>
  inline T apply(T val1, T val2) {
    return val1 - val2;
  }
};

struct Mul {
  template <typename T>
  T apply(T val1, T val2) {
    return val1 * val2;
  }
};

struct Div {
  template <typename T>
  T apply(T val1, T val2) {
    return val1 / val2;
  }
};

/**
 * Binary Elementwise Operation Functor.
 */

template <typename T, typename Op, typename Index, int VectorWidth>
class BinaryOp {
  using DataT = typename helpers::VectorType<T, VectorWidth>::type;
  using Load = helpers::io::Load<DataT>;
  using Store = helpers::io::Store<DataT>;

  ReadAccessor<T const> lhs_, rhs_;
  WriteAccessor<T> out_data_;
  const Index n_iterations_, n_offset_;

 public:
  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) {
    Index idx1 = item.get_id(0) * VectorWidth;
    Index idx2 = idx1;

    const auto in1 = lhs_.get_pointer();
    const auto in2 = rhs_.get_pointer();
    auto out = out_data_.get_pointer();

    Op op;
    auto rhs_val = Load()(in2, idx2);

    for (Index i = 0; i < n_iterations_; i++) {
      auto lhs_val = Load()(in1, idx1);
      auto out_val = op.apply(lhs_val, rhs_val);
      Store()(out, idx1, out_val);
      idx1 += n_offset_;
    }
  }

  BinaryOp(ReadAccessor<T const> lhs, ReadAccessor<T const> rhs,
           WriteAccessor<T> out_data)
      : lhs_(lhs),
        rhs_(rhs),
        out_data_(out_data),
        n_iterations_(static_cast<Index>(lhs.get_extent() / rhs.get_extent())),
        n_offset_(static_cast<Index>(rhs.get_extent())) {}
};

}  // namespace binaryop
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_BINARYOP_KERNELS_H_
