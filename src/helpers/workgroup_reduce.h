/*
 * Copyright 2018 Codeplay Software Ltd
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
#ifndef SYCLDNN_SRC_HELPERS_WORKGROUP_REDUCE_H_
#define SYCLDNN_SRC_HELPERS_WORKGROUP_REDUCE_H_

#include "src/helpers/tensor_index.h"
#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"
#include "src/helpers/window_index.h"

#include "sycldnn/accessor_types.h"
#include "sycldnn/conv2d/conv_type.h"
#include "sycldnn/depthwise_conv2d/params.h"
#include "sycldnn/helpers/macros.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace helpers {
namespace reduce {

namespace internal {

template <typename Index>
inline SNN_ALWAYS_INLINE Index
get_flattened_local_range(cl::sycl::nd_item<1> item) {
  return item.get_local_range(0);
}

template <typename Index>
inline SNN_ALWAYS_INLINE Index
get_flattened_local_range(cl::sycl::nd_item<2> item) {
  Index range_0 = item.get_local_range(0);
  Index range_1 = item.get_local_range(1);
  return range_0 * range_1;
}

template <typename Index>
inline SNN_ALWAYS_INLINE Index
get_flattened_local_range(cl::sycl::nd_item<3> item) {
  Index range_0 = item.get_local_range(0);
  Index range_1 = item.get_local_range(1);
  Index range_2 = item.get_local_range(2);
  return range_0 * range_1 * range_2;
}

template <typename Index>
inline SNN_ALWAYS_INLINE Index
get_flattened_local_id(cl::sycl::nd_item<1> item) {
  return item.get_local_id(0);
}

template <typename Index>
inline SNN_ALWAYS_INLINE Index
get_flattened_local_id(cl::sycl::nd_item<2> item) {
  Index id_0 = item.get_local_id(0);
  Index id_1 = item.get_local_id(1);
  Index range_0 = item.get_local_range(0);
  return id_1 * range_0 + id_0;
}

template <typename Index>
inline SNN_ALWAYS_INLINE Index
get_flattened_local_id(cl::sycl::nd_item<3> item) {
  Index id_0 = item.get_local_id(0);
  Index id_1 = item.get_local_id(1);
  Index id_2 = item.get_local_id(2);

  Index range_0 = item.get_local_range(0);
  Index range_1 = item.get_local_range(1);

  return (id_2 * range_1 + id_1) * range_0 + id_0;
}

/**
 * Conversion struct to change from a SYCL address space to a memory fence
 * space.
 */
template <cl::sycl::access::address_space Space>
struct AddressSpaceToFenceSpace {
  static constexpr auto FenceSpace =
      cl::sycl::access::fence_space::global_and_local;
};

template <>
struct AddressSpaceToFenceSpace<cl::sycl::access::address_space::local_space> {
  static constexpr auto FenceSpace = cl::sycl::access::fence_space::local_space;
};

}  // namespace internal

struct Sum {
  template <typename T>
  SNN_ALWAYS_INLINE T operator()(T lhs, T rhs) {
    return lhs + rhs;
  }
};

/**
 * Reduce a value across the workgroup.
 *
 * The final value is not broadcast across the workgroup, so the correct output
 * value is only returned to the work item with local id (0, 0, 0). If the value
 * is needed by all items in a workgroup this must be broadcast across the group
 * separately.
 *
 * Assumes that the workgroup range is a power of two.
 * Assumes that the workspace pointer has sufficient storage to hold
 * (workgroup_size / 2) DataType elements.
 */
template <typename Op, typename Index, typename DataType, int Dimensions,
          cl::sycl::access::address_space Space>
inline SNN_ALWAYS_INLINE DataType
workgroup_reduce(DataType value, cl::sycl::nd_item<Dimensions> item,
                 cl::sycl::multi_ptr<DataType, Space> workspace) {
  static_assert(Space != cl::sycl::access::address_space::constant_space,
                "Cannot use constant memory as workspace in a reduction.");
  static_assert(Space != cl::sycl::access::address_space::private_space,
                "Cannot use private memory as workspace in a reduction.");
  constexpr auto fence_space =
      internal::AddressSpaceToFenceSpace<Space>::FenceSpace;

  using Store = helpers::io::Store<DataType>;
  using Load = helpers::io::Load<DataType>;

  auto reduction_idx = internal::get_flattened_local_range<Index>(item);
  auto local_idx = internal::get_flattened_local_id<Index>(item);
  bool written = false;

  while (reduction_idx > 1) {
    reduction_idx /= 2;

    if (local_idx >= reduction_idx && !written) {
      Store()(workspace, local_idx, value);
      written = true;
    }

    item.barrier(fence_space);

    if (local_idx < reduction_idx) {
      DataType other_thread_val = Load()(workspace, local_idx + reduction_idx);
      value = Op()(value, other_thread_val);
    }
  }
  return value;
}

}  // namespace reduce
}  // namespace helpers
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_HELPERS_WORKGROUP_REDUCE_H_
