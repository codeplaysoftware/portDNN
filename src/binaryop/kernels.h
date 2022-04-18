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

 public:
  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<2> item) {
    Index rhs_idx = static_cast<Index>(item.get_id(1) * VectorWidth);
    Index lhs_idx =
        static_cast<Index>(item.get_id(0) * rhs_.get_extent()) + rhs_idx;

    const auto lhs = lhs_.get_pointer();
    const auto rhs = rhs_.get_pointer();
    auto out = out_data_.get_pointer();

    Op op;
    auto lhs_val = Load()(lhs, lhs_idx);
    auto rhs_val = Load()(rhs, rhs_idx);
    auto out_val = op.apply(lhs_val, rhs_val);
    Store()(out, lhs_idx, out_val);
  }

  BinaryOp(ReadAccessor<T const> lhs, ReadAccessor<T const> rhs,
           WriteAccessor<T> out_data)
      : lhs_(lhs), rhs_(rhs), out_data_(out_data) {}
};

template <typename T, typename Index, int VectorWidth>
class BinaryOp<T, internal::SoftmaxSub, Index, VectorWidth> {
  using DataT = typename helpers::VectorType<T, VectorWidth>::type;
  using ScalarDataT = typename helpers::VectorType<T, 1>::type;
  using Load = helpers::io::Load<DataT>;
  using ScalarLoad = helpers::io::Load<ScalarDataT>;
  using Store = helpers::io::Store<DataT>;

  ReadAccessor<T const> lhs_, rhs_;
  WriteAccessor<T> out_data_;
  const Index n_offset_, n_iterations_;

 public:
  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<2> item) {
    Index lhs_idx = static_cast<Index>(item.get_id(1) * VectorWidth);

    const auto lhs = lhs_.get_pointer();
    const auto rhs = rhs_.get_pointer();
    auto out = out_data_.get_pointer();

    Sub op;

    for (Index i = 0; i < n_iterations_; i++) {
      auto lhs_val = Load()(lhs, lhs_idx);
      auto rhs_val = ScalarLoad()(rhs, i);
      auto out_val = op.apply(lhs_val, static_cast<DataT>(rhs_val));
      Store()(out, lhs_idx, out_val);
      lhs_idx += n_offset_;
    }
  }

  BinaryOp(ReadAccessor<T const> lhs, ReadAccessor<T const> rhs,
           WriteAccessor<T> out_data)
      : lhs_(lhs),
        rhs_(rhs),
        out_data_(out_data),
        n_offset_(static_cast<Index>(lhs.get_extent() / rhs.get_extent())),
        n_iterations_(static_cast<Index>(rhs.get_extent())) {}
};

template <typename T, typename Index, int VectorWidth>
class BinaryOp<T, internal::SoftmaxDiv, Index, VectorWidth> {
  using DataT = typename helpers::VectorType<T, VectorWidth>::type;
  using ScalarDataT = typename helpers::VectorType<T, 1>::type;
  using Load = helpers::io::Load<DataT>;
  using ScalarLoad = helpers::io::Load<ScalarDataT>;
  using Store = helpers::io::Store<DataT>;

  ReadAccessor<T const> lhs_, rhs_;
  WriteAccessor<T> out_data_;
  const Index n_offset_, n_iterations_;

 public:
  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<2> item) {
    Index lhs_idx = static_cast<Index>(item.get_id(1) * VectorWidth);

    const auto lhs = lhs_.get_pointer();
    const auto rhs = rhs_.get_pointer();
    auto out = out_data_.get_pointer();

    Div op;

    for (Index i = 0; i < n_iterations_; i++) {
      auto lhs_val = Load()(lhs, lhs_idx);
      auto rhs_val = ScalarLoad()(rhs, i);
      auto out_val = op.apply(lhs_val, static_cast<DataT>(rhs_val));
      Store()(out, lhs_idx, out_val);
      lhs_idx += n_offset_;
    }
  }

  BinaryOp(ReadAccessor<T const> lhs, ReadAccessor<T const> rhs,
           WriteAccessor<T> out_data)
      : lhs_(lhs),
        rhs_(rhs),
        out_data_(out_data),
        n_offset_(static_cast<Index>(lhs.get_extent() / rhs.get_extent())),
        n_iterations_(static_cast<Index>(rhs.get_extent())) {}
};

}  // namespace binaryop
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_BINARYOP_KERNELS_H_
