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
#ifndef SYCLDNN_INCLUDE_INTERNAL_TRANSPOSE_LAUNCH_H_
#define SYCLDNN_INCLUDE_INTERNAL_TRANSPOSE_LAUNCH_H_

#include <CL/sycl.hpp>

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include <vector>

#include "sycldnn/export.h"

namespace sycldnn {

const std::vector<int> NHWC_TO_NCHW{0, 3, 1, 2};
const std::vector<int> NCHW_TO_NHWC{0, 2, 3, 1};

namespace transpose {
namespace internal {

/**
 * The internal tensor transpose launcher.
 *
 * Implemented in the compiled SYCL DNN library.
 */
template <typename T>
SNN_EXPORT SNNStatus launch_impl(BaseMemObject<T const>& input,
                                 BaseMemObject<T>& output,
                                 std::vector<int> dimensions,
                                 std::vector<int> permutation,
                                 cl::sycl::queue& queue);

/**
 * Internal tensor transpose launcher that is able to cast tensor types to the
 * implemented types.
 */
template <typename SrcT, typename DstT>
SNNStatus launch_cast(BaseMemObject<SrcT const>& input,
                      BaseMemObject<SrcT>& output, std::vector<int> dimensions,
                      std::vector<int> permutation, cl::sycl::queue& queue) {
  if (std::is_same<SrcT, DstT>::value) {
    return launch_impl(input, output, dimensions, permutation, queue);
  }
  auto& input_mem =
      dynamic_cast<MemObject<SrcT const, cl::sycl::buffer_allocator>&>(input);
  auto& output_mem =
      dynamic_cast<MemObject<SrcT, cl::sycl::buffer_allocator>&>(output);
  auto input_int_mem = input_mem.template cast<DstT>();
  auto output_int_mem = output_mem.template cast<DstT>();
  return launch_impl(input_int_mem, output_int_mem, dimensions, permutation,
                     queue);
}

#define SNN_LAUNCH_CAST(DST_T)                                                \
  template <typename T, typename std::enable_if<sizeof(T) == sizeof(DST_T),   \
                                                int>::type = 0>               \
  SNNStatus launch(BaseMemObject<T const>& input, BaseMemObject<T>& output,   \
                   std::vector<int> dimensions, std::vector<int> permutation, \
                   cl::sycl::queue& queue) {                                  \
    return launch_cast<T, DST_T>(input, output, dimensions, permutation,      \
                                 queue);                                      \
  }

SNN_LAUNCH_CAST(uint8_t);
SNN_LAUNCH_CAST(uint16_t);
SNN_LAUNCH_CAST(uint32_t);
SNN_LAUNCH_CAST(uint64_t);

#undef SNN_LAUNCH_CAST

}  // namespace internal
}  // namespace transpose
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_INTERNAL_TRANSPOSE_LAUNCH_H_
