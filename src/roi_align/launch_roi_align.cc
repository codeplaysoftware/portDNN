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
#ifndef SYCLDNN_SRC_ROI_ALIGN_LAUNCH_ROI_ALIGN_H_
#define SYCLDNN_SRC_ROI_ALIGN_LAUNCH_ROI_ALIGN_H_

#include "sycldnn/mem_object.h"

#include "sycldnn/roi_align/operators.h"
#include "sycldnn/roi_align/params.h"

#include "sycldnn/internal/roi_align/launch_internal.h"

#include "src/roi_align/queue_roi_align_kernel.h"

#include <CL/sycl.hpp>

#include "sycldnn/export.h"

namespace sycldnn {
namespace roi_align {
namespace internal {

template <typename T, typename BatchIndicesT, typename Index,
          template <typename> class PoolType>
SNNStatus launch_with_index(BaseMemObject<T const>& input,
                            BaseMemObject<T const>& rois,
                            BaseMemObject<BatchIndicesT const>& batch_indices,
                            BaseMemObject<T>& output, const RoiAlignParams& rap,
                            size_t threads, cl::sycl::queue& queue) {
  return queue_roi_align<T, BatchIndicesT, Index, PoolType>(
      input, rois, batch_indices, output, rap, threads, queue);
}

template <typename T, typename BatchIndicesT,
          template <typename> class PoolType>
SNNStatus launch_roi_align(BaseMemObject<T const>& input,
                           BaseMemObject<T const>& rois,
                           BaseMemObject<BatchIndicesT const>& batch_indices,
                           BaseMemObject<T>& output, const RoiAlignParams& rap,
                           cl::sycl::queue& queue) {
  const size_t threads = output.get_count();
  if (threads > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
#ifdef SNN_USE_INT64
    return launch_with_index<T, BatchIndicesT, int64_t, PoolType>(
        input, rois, batch_indices, output, rap, threads, queue);
#else
    return StatusCode::IndexExceeded;
#endif  // SNN_USE_INT64
  } else {
    return launch_with_index<T, BatchIndicesT, int32_t, PoolType>(
        input, rois, batch_indices, output, rap, threads, queue);
  }
}

#define INSTANTIATE_LAUNCH(DTYPE, BITYPE, OP)                                \
  template SNN_EXPORT SNNStatus launch_roi_align<DTYPE, BITYPE, OP>(         \
      BaseMemObject<DTYPE const> & input, BaseMemObject<DTYPE const> & rois, \
      BaseMemObject<BITYPE const> & batch_indices,                           \
      BaseMemObject<DTYPE> & output, const RoiAlignParams& rap,              \
      cl::sycl::queue& queue)

#define INSTANTIATE_FOR_TYPE(DTYPE, BI_TYPE)        \
  INSTANTIATE_LAUNCH(DTYPE, BI_TYPE, pooling::Max); \
  INSTANTIATE_LAUNCH(DTYPE, BI_TYPE, pooling::Average)

INSTANTIATE_FOR_TYPE(float, int32_t);
INSTANTIATE_FOR_TYPE(float, int64_t);

#ifdef SNN_USE_HALF
INSTANTIATE_FOR_TYPE(cl::sycl::half, int32_t);
INSTANTIATE_FOR_TYPE(cl::sycl::half, int64_t);
#endif  // SNN_USE_HALF

#ifdef SNN_USE_DOUBLE
INSTANTIATE_FOR_TYPE(double, int32_t);
INSTANTIATE_FOR_TYPE(double, int64_t);
#endif  // SNN_USE_DOUBLE

#undef INSTANTIATE_FOR_TYPE
#undef INSTANTIATE_LAUNCH

}  // namespace internal
}  // namespace roi_align
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_ROI_ALIGN_LAUNCH_ROI_ALIGN_H_
