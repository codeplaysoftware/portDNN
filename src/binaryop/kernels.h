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

#ifndef PORTDNN_SRC_BINARYOP_KERNELS_H_
#define PORTDNN_SRC_BINARYOP_KERNELS_H_

#include <CL/sycl.hpp>
#include <array>

#include "src/helpers/tensor_index.h"
#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"

#include "portdnn/accessor_types.h"

#include "portdnn/binaryop/operators.h"
#include "portdnn/binaryop/params.h"
#include "portdnn/helpers/dims.h"

namespace sycldnn {
namespace binaryop {

struct Add {
  template <typename T>
  inline T operator()(T lhs, T rhs) {
    return lhs + rhs;
  }
};

struct Sub {
  template <typename T>
  inline T operator()(T lhs, T rhs) {
    return lhs - rhs;
  }
};

struct Mul {
  template <typename T>
  T operator()(T lhs, T rhs) {
    return lhs * rhs;
  }
};

struct Div {
  template <typename T>
  T operator()(T lhs, T rhs) {
    return lhs / rhs;
  }
};

/**
 * Binary Elementwise Operation Functors.
 */

/**
 * Generic scalar vector. Any dimension can be broadcasted and at least one
 * dimension is broadcasted.
 */
template <typename T, typename Op, typename Index, bool IsUSM>
class BinaryOp {
  ReadMem<T const, IsUSM> lhs_, rhs_;
  WriteMem<T, IsUSM> out_;
  std::array<Index, MAX_DIMS> lhs_dims_;
  std::array<Index, MAX_DIMS> rhs_dims_;
  std::array<Index, MAX_DIMS> out_dims_;

 public:
  BinaryOp(ReadMem<T const, IsUSM> lhs, ReadMem<T const, IsUSM> rhs,
           WriteMem<T, IsUSM> out, const std::vector<Index>& lhs_dims,
           const std::vector<Index>& rhs_dims,
           const std::vector<Index>& out_dims)
      : lhs_(lhs), rhs_(rhs), out_(out) {
    size_t i = 0;
    for (; i < MAX_DIMS - lhs_dims.size(); ++i) {
      lhs_dims_[i] = 1;
      rhs_dims_[i] = 1;
      out_dims_[i] = 1;
    }
    for (size_t j = 0; i < MAX_DIMS; ++i, ++j) {
      lhs_dims_[i] = lhs_dims[j];
      rhs_dims_[i] = rhs_dims[j];
      out_dims_[i] = out_dims[j];
    }
  }

  cl::sycl::range<1> get_range() {
    return {size_t(helpers::get_total_size(out_dims_))};
  }

  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) const {
    Index out_idx = item.get_id(0);
    Index lhs_idx = 0;
    Index rhs_idx = 0;
    Index cumulative_lhs_stride = 1;
    Index cumulative_rhs_stride = 1;
    Index out_idx_remainder = out_idx;

    // Compute lhs and rhs ids. An id is added only if the corresponding
    // dimension is not broadcasted.
    // The dimensions are checked outside of the kernel.
    auto clamp_idx = [&](int i, Index unflatten_idx) {
      if (unflatten_idx < lhs_dims_[i]) {
        lhs_idx += unflatten_idx * cumulative_lhs_stride;
      }
      if (unflatten_idx < rhs_dims_[i]) {
        rhs_idx += unflatten_idx * cumulative_rhs_stride;
      }
    };
    for (int i = MAX_DIMS - 1; i >= 1; --i) {
      clamp_idx(i, out_idx_remainder % out_dims_[i]);
      cumulative_lhs_stride *= lhs_dims_[i];
      cumulative_rhs_stride *= rhs_dims_[i];
      out_idx_remainder /= out_dims_[i];
    }
    clamp_idx(0, out_idx_remainder);

    const auto lhs = lhs_.get_pointer().get();
    const auto rhs = rhs_.get_pointer().get();
    auto out = out_.get_pointer().get();

    Op op;
    out[out_idx] = op(lhs[lhs_idx], rhs[rhs_idx]);
  }
};

/**
 * 1D kernel with no broadcast.
 */
template <typename T, typename Op, typename Index, int VectorWidth, bool IsUSM>
class BinaryOpVec {
  using DataT = typename helpers::VectorType<T, VectorWidth>::type;
  using Load = helpers::io::Load<DataT>;
  using Store = helpers::io::Store<DataT>;

  ReadMem<T const, IsUSM> lhs_, rhs_;
  WriteMem<T, IsUSM> out_;
  const Index size;

 public:
  BinaryOpVec(ReadMem<T const, IsUSM> lhs, ReadMem<T const, IsUSM> rhs,
              WriteMem<T, IsUSM> out, const std::vector<Index>&,
              const std::vector<Index>&, const std::vector<Index>&)
      : lhs_(lhs), rhs_(rhs), out_(out), size(out.get_extent()) {}

  cl::sycl::range<1> get_range() { return {size_t(size / VectorWidth)}; }

  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<1> item) const {
    Index idx = item.get_id(0) * VectorWidth;

    const auto lhs = lhs_.get_pointer();
    const auto rhs = rhs_.get_pointer();
    auto out = out_.get_pointer();

    Op op;
    auto lhs_val = Load()(lhs, idx);
    auto rhs_val = Load()(rhs, idx);
    Store()(out, idx, op(lhs_val, rhs_val));
  }
};

/**
 * 2D kernel where the last lhs dimension is broadcasted.
 */
template <typename T, typename Op, typename Index, int VectorWidth, bool IsUSM>
class BinaryOpBcastLhsVec2D {
  using DataT = typename helpers::VectorType<T, VectorWidth>::type;
  using Load = helpers::io::Load<DataT>;
  using Store = helpers::io::Store<DataT>;

  ReadMem<T const, IsUSM> lhs_, rhs_;
  WriteMem<T, IsUSM> out_;
  const std::array<Index, 2> out_dims_;

 public:
  BinaryOpBcastLhsVec2D(ReadMem<T const, IsUSM> lhs,
                        ReadMem<T const, IsUSM> rhs, WriteMem<T, IsUSM> out,
                        const std::vector<Index>&, const std::vector<Index>&,
                        const std::vector<Index>& out_dims)
      : lhs_(lhs), rhs_(rhs), out_(out), out_dims_{out_dims[0], out_dims[1]} {}

  cl::sycl::range<2> get_range() {
    return {size_t(out_dims_[0]), size_t(out_dims_[1] / VectorWidth)};
  }

  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<2> item) const {
    Index batch = item.get_id(0);
    Index inner = item.get_id(1);
    Index out_idx = batch * out_dims_[1] + inner * VectorWidth;

    const auto lhs = lhs_.get_pointer().get();
    const auto rhs = rhs_.get_pointer();
    auto out = out_.get_pointer();

    Op op;
    auto lhs_val = DataT(lhs[batch]);
    auto rhs_val = Load()(rhs, out_idx);
    Store()(out, out_idx, op(lhs_val, rhs_val));
  }
};

/**
 * 2D kernel where the last rhs dimension is broadcasted.
 */
template <typename T, typename Op, typename Index, int VectorWidth, bool IsUSM>
class BinaryOpBcastRhsVec2D {
  using DataT = typename helpers::VectorType<T, VectorWidth>::type;
  using Load = helpers::io::Load<DataT>;
  using Store = helpers::io::Store<DataT>;

  ReadMem<T const, IsUSM> lhs_, rhs_;
  WriteMem<T, IsUSM> out_;
  const std::array<Index, 2> out_dims_;

 public:
  BinaryOpBcastRhsVec2D(ReadMem<T const, IsUSM> lhs,
                        ReadMem<T const, IsUSM> rhs, WriteMem<T, IsUSM> out,
                        const std::vector<Index>&, const std::vector<Index>&,
                        const std::vector<Index>& out_dims)
      : lhs_(lhs), rhs_(rhs), out_(out), out_dims_{out_dims[0], out_dims[1]} {}

  cl::sycl::range<2> get_range() {
    return {size_t(out_dims_[0]), size_t(out_dims_[1] / VectorWidth)};
  }

  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<2> item) const {
    Index batch = item.get_id(0);
    Index inner = item.get_id(1);
    Index out_idx = batch * out_dims_[1] + inner * VectorWidth;

    const auto lhs = lhs_.get_pointer();
    const auto rhs = rhs_.get_pointer().get();
    auto out = out_.get_pointer();

    Op op;
    auto lhs_val = Load()(lhs, out_idx);
    auto rhs_val = DataT(rhs[batch]);
    Store()(out, out_idx, op(lhs_val, rhs_val));
  }
};

/**
 * 3D kernel where the outer lhs dimension is broadcasted
 * (in [batch, outer, inner])
 */
template <typename T, typename Op, typename Index, int VectorWidth, bool IsUSM>
class BinaryOpBcastLhsVec3D {
  using DataT = typename helpers::VectorType<T, VectorWidth>::type;
  using Load = helpers::io::Load<DataT>;
  using Store = helpers::io::Store<DataT>;

  ReadMem<T const, IsUSM> lhs_, rhs_;
  WriteMem<T, IsUSM> out_;
  const std::array<Index, 3> out_dims_;

 public:
  BinaryOpBcastLhsVec3D(ReadMem<T const, IsUSM> lhs,
                        ReadMem<T const, IsUSM> rhs, WriteMem<T, IsUSM> out,
                        const std::vector<Index>&, const std::vector<Index>&,
                        const std::vector<Index>& out_dims)
      : lhs_(lhs),
        rhs_(rhs),
        out_(out),
        out_dims_{out_dims[0], out_dims[1], out_dims[2]} {}

  cl::sycl::range<3> get_range() {
    return {size_t(out_dims_[0]), size_t(out_dims_[1]),
            size_t(out_dims_[2] / VectorWidth)};
  }

  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<3> item) const {
    Index batch = item.get_id(0);
    Index outer = item.get_id(1);
    Index inner = item.get_id(2);
    Index out_idx =
        (batch * out_dims_[1] + outer) * out_dims_[2] + inner * VectorWidth;
    Index lhs_idx = batch * out_dims_[2] + inner * VectorWidth;

    const auto lhs = lhs_.get_pointer();
    const auto rhs = rhs_.get_pointer();
    auto out = out_.get_pointer();

    Op op;
    auto lhs_val = Load()(lhs, lhs_idx);
    auto rhs_val = Load()(rhs, out_idx);
    Store()(out, out_idx, op(lhs_val, rhs_val));
  }
};

/**
 * 3D kernel where the outer rhs dimension is broadcasted
 * (in [batch, outer, inner])
 */
template <typename T, typename Op, typename Index, int VectorWidth, bool IsUSM>
class BinaryOpBcastRhsVec3D {
  using DataT = typename helpers::VectorType<T, VectorWidth>::type;
  using Load = helpers::io::Load<DataT>;
  using Store = helpers::io::Store<DataT>;

  ReadMem<T const, IsUSM> lhs_, rhs_;
  WriteMem<T, IsUSM> out_;
  const std::array<Index, 3> out_dims_;

 public:
  BinaryOpBcastRhsVec3D(ReadMem<T const, IsUSM> lhs,
                        ReadMem<T const, IsUSM> rhs, WriteMem<T, IsUSM> out,
                        const std::vector<Index>&, const std::vector<Index>&,
                        const std::vector<Index>& out_dims)
      : lhs_(lhs),
        rhs_(rhs),
        out_(out),
        out_dims_{out_dims[0], out_dims[1], out_dims[2]} {}

  cl::sycl::range<3> get_range() {
    return {size_t(out_dims_[0]), size_t(out_dims_[1]),
            size_t(out_dims_[2] / VectorWidth)};
  }

  SNN_ALWAYS_INLINE void operator()(cl::sycl::item<3> item) const {
    Index batch = item.get_id(0);
    Index outer = item.get_id(1);
    Index inner = item.get_id(2);
    Index out_idx =
        (batch * out_dims_[1] + outer) * out_dims_[2] + inner * VectorWidth;
    Index rhs_idx = batch * out_dims_[2] + inner * VectorWidth;

    const auto lhs = lhs_.get_pointer();
    const auto rhs = rhs_.get_pointer();
    auto out = out_.get_pointer();

    Op op;
    auto lhs_val = Load()(lhs, out_idx);
    auto rhs_val = Load()(rhs, rhs_idx);
    Store()(out, out_idx, op(lhs_val, rhs_val));
  }
};

}  // namespace binaryop
}  // namespace sycldnn

#endif  // PORTDNN_SRC_BINARYOP_KERNELS_H_
