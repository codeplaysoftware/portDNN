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

#ifndef SYCLDNN_INCLUDE_INTERNAL_SCATTER_ND_LAUNCH_INTERNAL_H_
#define SYCLDNN_INCLUDE_INTERNAL_SCATTER_ND_LAUNCH_INTERNAL_H_

#include "sycldnn/helpers/sycl_language_helpers.h"
#include "sycldnn/status.h"

#include "sycldnn/export.h"

#include "sycldnn/scatter_nd/operators.h"
#include "sycldnn/scatter_nd/sizes.h"
namespace sycldnn {
namespace scatter_nd {
namespace internal {

/**
 * The internal scatter_nd launcher.
 *
 */
template <typename T, typename Index, typename ScatterNDType, int IndexDepth>
SNN_EXPORT SNNStatus launch_scatter_nd(BaseMemObject<T const>& input,
                                       BaseMemObject<Index const>& indices,
                                       BaseMemObject<T const>& update,
                                       BaseMemObject<T>& output,
                                       const ScatterNDSizes& sizes,
                                       cl::sycl::queue& queue);

/**
 * Internal scatter_nd launcher that casts tensor types to the
 * implemented types when needed.
 */
template <typename SrcT, typename DstT, typename Index, typename ScatterNDType,
          int IndexDepth>
SNNStatus launch_cast(BaseMemObject<SrcT const>& input,
                      BaseMemObject<Index const>& indices,
                      BaseMemObject<SrcT const>& updates,
                      BaseMemObject<SrcT>& output, const ScatterNDSizes& sizes,
                      cl::sycl::queue& queue) {
  if (std::is_same<SrcT, DstT>::value) {
    return launch_scatter_nd<SrcT, Index, ScatterNDType, IndexDepth>(
        input, indices, updates, output, sizes, queue);
  }
  if (!std::is_same<ScatterNDType, Assign>::value) {
    return launch_scatter_nd<SrcT, Index, ScatterNDType, IndexDepth>(
        input, indices, updates, output, sizes, queue);
  }
  auto& input_mem = dynamic_cast<
      MemObject<SrcT const, sycldnn::helpers::buffer_allocator<SrcT>>&>(input);
  auto& updates_mem = dynamic_cast<
      MemObject<SrcT const, sycldnn::helpers::buffer_allocator<SrcT>>&>(
      updates);
  auto& output_mem =
      dynamic_cast<MemObject<SrcT, sycldnn::helpers::buffer_allocator<SrcT>>&>(
          output);
  auto input_cast_mem = input_mem.template cast<DstT const>();
  auto updates_cast_mem = updates_mem.template cast<DstT const>();
  auto output_cast_mem = output_mem.template cast<DstT>();
  return launch_scatter_nd<DstT, Index, ScatterNDType, IndexDepth>(
      input_cast_mem, indices, updates_cast_mem, output_cast_mem, sizes, queue);
}

#define SNN_LAUNCH_CAST(DST_T)                                                \
  template <                                                                  \
      typename T, typename Index, typename ScatterNDType, int IndexDepth,     \
      typename std::enable_if<sizeof(T) == sizeof(DST_T), int>::type = 0>     \
  SNNStatus launch(BaseMemObject<T const>& input,                             \
                   BaseMemObject<Index const>& indices,                       \
                   BaseMemObject<T const>& updates, BaseMemObject<T>& output, \
                   const ScatterNDSizes& sizes, cl::sycl::queue& queue) {     \
    return launch_cast<T, DST_T, Index, ScatterNDType, IndexDepth>(           \
        input, indices, updates, output, sizes, queue);                       \
  }

SNN_LAUNCH_CAST(uint8_t);
SNN_LAUNCH_CAST(uint16_t);
SNN_LAUNCH_CAST(uint32_t);
SNN_LAUNCH_CAST(uint64_t);

}  // namespace internal
}  // namespace scatter_nd
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_INTERNAL_SCATTER_ND_LAUNCH_INTERNAL_H_
