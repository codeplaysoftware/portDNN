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
#ifndef PORTDNN_SRC_ROI_ALIGN_LAUNCH_ROI_ALIGN_H_
#define PORTDNN_SRC_ROI_ALIGN_LAUNCH_ROI_ALIGN_H_

#include "portdnn/mem_object.h"

#include "portdnn/roi_align/operators.h"
#include "portdnn/roi_align/params.h"

#include "portdnn/internal/roi_align/launch_internal.h"

#include "src/roi_align/queue_roi_align_kernel.h"

#include <CL/sycl.hpp>

#include "portdnn/export.h"

namespace sycldnn {
namespace roi_align {
namespace internal {

template <typename T, typename BatchIndicesT,
          template <typename> class PoolType, template <typename> class MemObj>
SNNStatus launch_roi_align(MemObj<T const>& input, MemObj<T const>& rois,
                           MemObj<BatchIndicesT const>& batch_indices,
                           MemObj<T>& output, const RoiAlignParams& rap,
                           cl::sycl::queue& queue,
                           const std::vector<cl::sycl::event>& events) {
  const size_t threads = output.get_extent();
  if (threads > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
#ifdef SNN_USE_INT64
    return queue_roi_align<T, BatchIndicesT, int64_t, PoolType>(
        input, rois, batch_indices, output, rap, threads, queue, events);
#else
    return StatusCode::IndexExceeded;
#endif  // SNN_USE_INT64
  } else {
    return queue_roi_align<T, BatchIndicesT, int32_t, PoolType>(
        input, rois, batch_indices, output, rap, threads, queue, events);
  }
}

#define INSTANTIATE_LAUNCH(DTYPE, BITYPE, OP, MEM_OBJ_TYPE)                \
  template SNN_EXPORT SNNStatus                                            \
  launch_roi_align<DTYPE, BITYPE, OP, MEM_OBJ_TYPE>(                       \
      MEM_OBJ_TYPE<DTYPE const> & input, MEM_OBJ_TYPE<DTYPE const> & rois, \
      MEM_OBJ_TYPE<BITYPE const> & batch_indices,                          \
      MEM_OBJ_TYPE<DTYPE> & output, const RoiAlignParams& rap,             \
      cl::sycl::queue& queue, const std::vector<cl::sycl::event>& events)

#define INSTANTIATE_FOR_DTYPE_AND_MEM_OBJ_TYPE(DTYPE, BI_TYPE, MEM_OBJ_TYPE) \
  INSTANTIATE_LAUNCH(DTYPE, BI_TYPE, MaxPool, MEM_OBJ_TYPE);                 \
  INSTANTIATE_LAUNCH(DTYPE, BI_TYPE, AveragePool, MEM_OBJ_TYPE)

INSTANTIATE_FOR_DTYPE_AND_MEM_OBJ_TYPE(float, int32_t, BufferMemObject);
INSTANTIATE_FOR_DTYPE_AND_MEM_OBJ_TYPE(float, int64_t, BufferMemObject);

#ifdef SNN_ENABLE_USM
INSTANTIATE_FOR_DTYPE_AND_MEM_OBJ_TYPE(float, int32_t, USMMemObject);
INSTANTIATE_FOR_DTYPE_AND_MEM_OBJ_TYPE(float, int64_t, USMMemObject);
#endif  // SNN_ENABLE_USM

#ifdef SNN_USE_HALF
INSTANTIATE_FOR_DTYPE_AND_MEM_OBJ_TYPE(cl::sycl::half, int32_t,
                                       BufferMemObject);
INSTANTIATE_FOR_DTYPE_AND_MEM_OBJ_TYPE(cl::sycl::half, int64_t,
                                       BufferMemObject);
#ifdef SNN_ENABLE_USM
INSTANTIATE_FOR_DTYPE_AND_MEM_OBJ_TYPE(cl::sycl::half, int32_t, USMMemObject);
INSTANTIATE_FOR_DTYPE_AND_MEM_OBJ_TYPE(cl::sycl::half, int64_t, USMMemObject);
#endif  // SNN_ENABLE_USM
#endif  // SNN_USE_HALF

#ifdef SNN_USE_DOUBLE
INSTANTIATE_FOR_DTYPE_AND_MEM_OBJ_TYPE(double, int32_t, BufferMemObject);
INSTANTIATE_FOR_DTYPE_AND_MEM_OBJ_TYPE(double, int64_t, BufferMemObject);
#ifdef SNN_ENABLE_USM
INSTANTIATE_FOR_DTYPE_AND_MEM_OBJ_TYPE(double, int32_t, USMMemObject);
INSTANTIATE_FOR_DTYPE_AND_MEM_OBJ_TYPE(double, int64_t, USMMemObject);
#endif  // SNN_ENABLE_USM
#endif  // SNN_USE_DOUBLE

#undef INSTANTIATE_FOR_DTYPE_AND_MEM_OBJ_TYPE
#undef INSTANTIATE_LAUNCH

}  // namespace internal
}  // namespace roi_align
}  // namespace sycldnn

#endif  // PORTDNN_SRC_ROI_ALIGN_LAUNCH_ROI_ALIGN_H_
