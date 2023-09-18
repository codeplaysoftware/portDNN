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
#ifndef PORTDNN_SRC_HELPERS_FLATTENED_ID_H_
#define PORTDNN_SRC_HELPERS_FLATTENED_ID_H_

#include "portdnn/helpers/macros.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace helpers {

/** Get the total local workgroup size, flattened to one dimension. */
template <typename Index = size_t>
inline SNN_ALWAYS_INLINE Index
get_flattened_local_range(cl::sycl::nd_item<1> item) {
  return item.get_local_range(0);
}

/** \copydoc get_flattened_local_range */
template <typename Index = size_t>
inline SNN_ALWAYS_INLINE Index
get_flattened_local_range(cl::sycl::nd_item<2> item) {
  Index range_0 = item.get_local_range(0);
  Index range_1 = item.get_local_range(1);
  return range_0 * range_1;
}

/** \copydoc get_flattened_local_range */
template <typename Index = size_t>
inline SNN_ALWAYS_INLINE Index
get_flattened_local_range(cl::sycl::nd_item<3> item) {
  Index range_0 = item.get_local_range(0);
  Index range_1 = item.get_local_range(1);
  Index range_2 = item.get_local_range(2);
  return range_0 * range_1 * range_2;
}

/** Get the local thread id, flattened to one dimension. */
template <typename Index = size_t>
inline SNN_ALWAYS_INLINE Index
get_flattened_local_id(cl::sycl::nd_item<1> item) {
  return item.get_local_id(0);
}

/** \copydoc get_flattened_local_id */
template <typename Index = size_t>
inline SNN_ALWAYS_INLINE Index
get_flattened_local_id(cl::sycl::nd_item<2> item) {
  Index id_0 = item.get_local_id(0);
  Index id_1 = item.get_local_id(1);

  Index range_0 = item.get_local_range(0);

  return id_1 * range_0 + id_0;
}

/** \copydoc get_flattened_local_id */
template <typename Index = size_t>
inline SNN_ALWAYS_INLINE Index
get_flattened_local_id(cl::sycl::nd_item<3> item) {
  Index id_0 = item.get_local_id(0);
  Index id_1 = item.get_local_id(1);
  Index id_2 = item.get_local_id(2);

  Index range_0 = item.get_local_range(0);
  Index range_1 = item.get_local_range(1);

  return (id_2 * range_1 + id_1) * range_0 + id_0;
}

/** Get the global thread id, flattened to one dimension. */
template <typename Index = size_t>
inline SNN_ALWAYS_INLINE Index
get_flattened_global_id(cl::sycl::nd_item<1> item) {
  return item.get_global_id(0);
}

/** \copydoc get_flattened_global_id */
template <typename Index = size_t>
inline SNN_ALWAYS_INLINE Index
get_flattened_global_id(cl::sycl::nd_item<2> item) {
  Index id_0 = item.get_global_id(0);
  Index id_1 = item.get_global_id(1);

  Index range_0 = item.get_global_range(0);

  return id_1 * range_0 + id_0;
}

/** \copydoc get_flattened_global_id */
template <typename Index = size_t>
inline SNN_ALWAYS_INLINE Index
get_flattened_global_id(cl::sycl::nd_item<3> item) {
  Index id_0 = item.get_global_id(0);
  Index id_1 = item.get_global_id(1);
  Index id_2 = item.get_global_id(2);

  Index range_0 = item.get_global_range(0);
  Index range_1 = item.get_global_range(1);

  return (id_2 * range_1 + id_1) * range_0 + id_0;
}

/** Get the group id, flattened to one dimension. */
template <typename Index = size_t>
inline SNN_ALWAYS_INLINE Index
get_flattened_group_id(cl::sycl::nd_item<1> item) {
  return item.get_group(0);
}

/** \copydoc get_flattened_group_id */
template <typename Index = size_t>
inline SNN_ALWAYS_INLINE Index
get_flattened_group_id(cl::sycl::nd_item<2> item) {
  Index id_0 = item.get_group(0);
  Index id_1 = item.get_group(1);

  Index range_0 = item.get_group_range(0);

  return id_1 * range_0 + id_0;
}

/** \copydoc get_flattened_global_id */
template <typename Index = size_t>
inline SNN_ALWAYS_INLINE Index
get_flattened_group_id(cl::sycl::nd_item<3> item) {
  Index id_0 = item.get_group(0);
  Index id_1 = item.get_group(1);
  Index id_2 = item.get_group(2);

  Index range_0 = item.get_group_range(0);
  Index range_1 = item.get_group_range(1);

  return (id_2 * range_1 + id_1) * range_0 + id_0;
}

}  // namespace helpers
}  // namespace sycldnn

#endif  // PORTDNN_SRC_HELPERS_FLATTENED_ID_H_
