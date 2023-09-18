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

#ifndef PORTDNN_SRC_POINTWISE_KERNELS_H_
#define PORTDNN_SRC_POINTWISE_KERNELS_H_

#include <CL/sycl.hpp>

#include "portdnn/helpers/macros.h"

#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"

#include "portdnn/accessor_types.h"

#include "portdnn/helpers/minmax.h"

#include "portdnn/pointwise/direction.h"

namespace sycldnn {
namespace pointwise {

/**
 * Note that our implementation of Relu uses f'(x) = 0 for x = 0.
 */
template <typename Direction>
struct Relu {
  template <typename DType>
  DType apply(DType val);
  template <typename DType>
  DType apply(DType val, DType err);
};

template <>
template <typename DType>
DType Relu<Forward>::apply(DType val) {
  return cl::sycl::max(val, DType{0});
}

template <>
template <typename DType>
DType Relu<Gradient>::apply(DType val, DType err) {
  auto mask = cl::sycl::isgreater(val, DType{0.f});
  return cl::sycl::select(DType{0.f}, err, mask);
}

template <typename Direction>
struct Tanh {
  template <typename DType>
  DType apply(DType val);
  template <typename DType>
  DType apply(DType val, DType err);
};

template <>
template <typename DType>
DType Tanh<Forward>::apply(DType val) {
  return cl::sycl::tanh(val);
}

template <>
template <typename DType>
DType Tanh<Gradient>::apply(DType val, DType err) {
  return (DType{1} - (val * val)) * err;
}

template <typename Direction>
struct Exp {
  template <typename DType>
  DType apply(DType val);
  template <typename DType>
  DType apply(DType val, DType err);
};

template <>
template <typename DType>
DType Exp<Forward>::apply(DType val) {
  return cl::sycl::exp(val);
}

// TODO: correct the exp working in backprop | err might not be a relevant
// potential solution might involve PointwiseOp specialization
// for Exp and Backprop
template <>
template <typename DType>
DType Exp<Gradient>::apply(DType val, DType err) {
  return cl::sycl::exp(val) + err;
}

template <typename Direction>
struct Log {
  template <typename DType>
  DType apply(DType val);
  template <typename DType>
  DType apply(DType val, DType err);
};

template <>
template <typename DType>
DType Log<Forward>::apply(DType val) {
  return cl::sycl::log(val);
}

template <>
template <typename DType>
DType Log<Gradient>::apply(DType val, DType err) {
  return (DType{1} / val) * err;
}

template <typename Direction>
struct Floor {
  template <typename DType>
  DType apply(DType val);
};

template <>
template <typename DType>
DType Floor<Forward>::apply(DType val) {
  return cl::sycl::floor(val);
}

template <typename Direction>
struct Sqrt {
  template <typename DType>
  DType apply(DType val);
  template <typename DType>
  DType apply(DType val, DType err);
};

template <>
template <typename DType>
DType Sqrt<Forward>::apply(DType val) {
  return cl::sycl::sqrt(val);
}

template <>
template <typename DType>
DType Sqrt<Gradient>::apply(DType val, DType err) {
  return (DType{0.5} / val) * err;
}

template <typename T, typename Index, template <typename> class Op,
          typename Direction, int VectorWidth, bool IsUSM>
class PointwiseOp;

/**
 * Note that PointwiseOp writes to a new buffer for the forward pass,
 * since we wish to keep the results for the backpropagation stage
 * if we are training.
 */

template <typename T, typename Index, template <typename> class Op,
          int VectorWidth, bool IsUSM>
class PointwiseOp<T, Index, Op, Forward, VectorWidth, IsUSM> {
  using DataType = typename helpers::VectorType<T, VectorWidth>::type;
  using LoadData = helpers::io::Load<DataType>;
  using StoreData = helpers::io::Store<DataType>;

  ReadMem<T const, IsUSM> input_;
  WriteMem<T, IsUSM> output_;
  Index const n_items_;

 public:
  PointwiseOp(ReadMem<T const, IsUSM> const& input,
              WriteMem<T, IsUSM> const& output, Index const num_items)
      : input_{input}, output_{output}, n_items_{num_items} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<1> item) const {
    Index const idx = item.get_id(0);

    if (idx < n_items_) {
      Op<Forward> op;
      auto vec_idx = idx * VectorWidth;

      auto in_ptr = input_.get_pointer();
      auto out_ptr = output_.get_pointer();

      auto in_value = LoadData()(in_ptr, vec_idx);
      auto out_value = op.apply(in_value);
      StoreData()(out_ptr, vec_idx, out_value);
    }
  }
};

template <typename T, typename Index, template <typename> class Op,
          int VectorWidth, bool IsUSM>
class PointwiseOp<T, Index, Op, Gradient, VectorWidth, IsUSM> {
  using DataType = typename helpers::VectorType<T, VectorWidth>::type;
  using LoadData = helpers::io::Load<DataType>;
  using StoreData = helpers::io::Store<DataType>;

  ReadMem<T const, IsUSM> output_forward_;
  ReadMem<T const, IsUSM> input_backprop_;
  WriteMem<T, IsUSM> output_backprop_;
  Index const n_items_;

 public:
  PointwiseOp(ReadMem<T const, IsUSM> const& output_forward,
              ReadMem<T const, IsUSM> const& input_backprop,
              WriteMem<T, IsUSM> const& output_backprop, Index const num_items)
      : output_forward_{output_forward},
        input_backprop_{input_backprop},
        output_backprop_{output_backprop},
        n_items_{num_items} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<1> item) const {
    Index const idx = item.get_id(0);

    if (idx < n_items_) {
      Op<Gradient> op;
      auto vec_idx = idx * VectorWidth;

      auto out_fwd_ptr = output_forward_.get_pointer();
      auto in_bk_ptr = input_backprop_.get_pointer();
      auto out_bk_ptr = output_backprop_.get_pointer();

      auto out_fwd_value = LoadData()(out_fwd_ptr, vec_idx);
      auto in_bk_value = LoadData()(in_bk_ptr, vec_idx);

      auto out_bk_value = op.apply(out_fwd_value, in_bk_value);

      StoreData()(out_bk_ptr, vec_idx, out_bk_value);
    }
  }
};

}  // namespace pointwise
}  // namespace sycldnn

#endif  // PORTDNN_SRC_POINTWISE_KERNELS_H_
