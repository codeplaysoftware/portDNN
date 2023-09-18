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
#ifndef PORTDNN_SRC_REDUCE_SUBGROUP_KERNEL_H_
#define PORTDNN_SRC_REDUCE_SUBGROUP_KERNEL_H_

#include "portdnn/accessor_types.h"
#include "portdnn/reduce/operators.h"
#include "portdnn/status.h"

namespace sycldnn {
namespace reduce {
namespace internal {

template <typename T, typename Index, typename Op>
struct SubgroupReducer;

template <typename T, typename Index>
struct SubgroupReducer<T, Index, Add> {
  static constexpr bool RequireFinalize = false;

  SNN_ALWAYS_INLINE T reduce(cl::sycl::sub_group sub_group, T x) {
    return sub_group.reduce(x, cl::sycl::plus<T>());
  }

  SNN_ALWAYS_INLINE T finalize(T x, Index) { return x; }
};

template <typename T, typename Index>
struct SubgroupReducer<T, Index, Mean> {
  static constexpr bool RequireFinalize = true;

  SNN_ALWAYS_INLINE T reduce(cl::sycl::sub_group sub_group, T x) {
    return sub_group.reduce(x, cl::sycl::plus<T>());
  }

  SNN_ALWAYS_INLINE T finalize(T x, Index outer_size) { return x / outer_size; }
};

template <typename T, typename Index>
struct SubgroupReducer<T, Index, Max> {
  static constexpr bool RequireFinalize = false;

  SNN_ALWAYS_INLINE T reduce(cl::sycl::experimental::sub_group sub_group, T x) {
    return sub_group.reduce(x, cl::sycl::experimental::maximum<T>());
  }

  SNN_ALWAYS_INLINE T finalize(T x, Index) { return x; }
};

template <typename T, typename Index>
struct SubgroupReducer<T, Index, Min> {
  static constexpr bool RequireFinalize = false;

  SNN_ALWAYS_INLINE T reduce(cl::sycl::experimental::sub_group sub_group, T x) {
    return sub_group.reduce(x, cl::sycl::experimental::minimum<T>());
  }

  SNN_ALWAYS_INLINE T finalize(T x, Index) { return x; }
};

}  // namespace internal

template <typename T, typename Index, typename Op, bool IsUSM>
struct ReduceSubgroupKernel {
  ReduceSubgroupKernel(ReadMem<T const, IsUSM> const& input,
                       WriteMem<T, IsUSM> const& output, Index sub_group_size,
                       Index reduce_size, Index in_size1, Index out_size1)
      : input_{input},
        output_{output},
        sub_group_size_{sub_group_size},
        reduce_size_{reduce_size},
        in_size1_{in_size1},
        out_size1_{out_size1} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::nd_item<2> nd_item) {
    const auto input = input_.get_pointer();
    auto output = output_.get_pointer();
    auto sub_group = nd_item.get_sub_group();
    cl::sycl::id<2> id = nd_item.get_global_id();
    size_t in_id = id[0] * in_size1_ + id[1];
    size_t out_id = id[0] * out_size1_ + id[1] / sub_group_size_;
    T input_val = Index(id[1]) < reduce_size_ ? input[in_id] : 0;

    internal::SubgroupReducer<T, Index, Op> reducer;
    output[out_id] = reducer.reduce(sub_group, input_val);
  }

 private:
  ReadMem<T const, IsUSM> input_;
  WriteMem<T, IsUSM> output_;
  Index const sub_group_size_;
  Index const reduce_size_;
  Index const in_size1_;
  Index const out_size1_;
};

template <typename T, typename Index, typename Op, bool IsUSM>
struct ReduceFinalize {
  ReduceFinalize(ReadWriteMem<T, IsUSM> const& output, Index finalize_param)
      : output_{output}, finalize_param_{finalize_param} {}

  void SNN_ALWAYS_INLINE operator()(cl::sycl::item<1> item) {
    auto output = output_.get_pointer();
    internal::SubgroupReducer<T, Index, Op> reducer;
    auto& out = output[item.get_linear_id()];
    out = reducer.finalize(out, finalize_param_);
  }

 private:
  ReadWriteMem<T, IsUSM> output_;
  Index const finalize_param_;
};

}  // namespace reduce
}  // namespace sycldnn
#endif  // PORTDNN_SRC_REDUCE_SUBGROUP_KERNEL_H_
