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
#ifndef SYCLDNN_INCLUDE_INTERNAL_GATHER_LAUNCH_H_
#define SYCLDNN_INCLUDE_INTERNAL_GATHER_LAUNCH_H_

#include <CL/sycl.hpp>

#include "sycldnn/export.h"
#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/gather/sizes.h"

namespace sycldnn {
namespace gather {
namespace internal {

/**
 * The internal Gather launcher.
 * Implemented in the compiled SYCL DNN library.
 */
template <typename T, typename Index>
SNN_EXPORT SNNStatus launch_impl(BaseMemObject<T const>& input,
                                 BaseMemObject<Index const>& indices,
                                 BaseMemObject<T>& output,
                                 const GatherSizes& sizes,
                                 cl::sycl::queue& queue);

/**
 * Internal gather launcher that casts tensor types to the
 * implemented types when needed.
 */
template <typename SrcT, typename DstT, typename Index>
SNNStatus launch_cast(BaseMemObject<SrcT const>& input,
                      BaseMemObject<Index const>& indices,
                      BaseMemObject<SrcT>& output, const GatherSizes& sizes,
                      cl::sycl::queue& queue) {
  if (std::is_same<SrcT, DstT>::value) {
    return launch_impl(input, indices, output, sizes, queue);
  }
  auto& input_mem =
      dynamic_cast<MemObject<SrcT const, cl::sycl::buffer_allocator>&>(input);
  auto& output_mem =
      dynamic_cast<MemObject<SrcT, cl::sycl::buffer_allocator>&>(output);
  auto input_int_mem = input_mem.template cast<DstT>();
  auto output_int_mem = output_mem.template cast<DstT>();
  return launch_impl(input_int_mem, indices, output_int_mem, sizes, queue);
}

#define SNN_LAUNCH_CAST(DST_T)                                                 \
  template <typename T, typename Index,                                        \
            typename std::enable_if<sizeof(T) == sizeof(DST_T), int>::type =   \
                0>                                                             \
  SNNStatus launch(BaseMemObject<T const>& input,                              \
                   BaseMemObject<Index const>& indices,                        \
                   BaseMemObject<T>& output, const GatherSizes& sizes,         \
                   cl::sycl::queue& queue) {                                   \
    return launch_cast<T, DST_T, Index>(input, indices, output, sizes, queue); \
  }

SNN_LAUNCH_CAST(uint8_t);
SNN_LAUNCH_CAST(uint16_t);
SNN_LAUNCH_CAST(uint32_t);
SNN_LAUNCH_CAST(uint64_t);

#undef SNN_LAUNCH_CAST

}  // namespace internal
}  // namespace gather
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_INTERNAL_GATHER_LAUNCH_H_
