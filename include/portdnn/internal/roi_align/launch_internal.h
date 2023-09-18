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

#ifndef PORTDNN_INCLUDE_ROI_ALIGN_LAUNCH_INTERNAL_H_
#define PORTDNN_INCLUDE_ROI_ALIGN_LAUNCH_INTERNAL_H_

#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/roi_align/params.h"

#include "portdnn/export.h"

namespace sycldnn {
namespace roi_align {
namespace internal {

template <typename T, typename Index, template <typename> class PoolType,
          template <typename> class MemObj>
SNN_EXPORT SNNStatus
launch_roi_align(MemObj<T const>& input, MemObj<T const>& rois,
                 MemObj<Index const>& batch_indices, MemObj<T>& output,
                 const RoiAlignParams& rap, cl::sycl::queue& queue,
                 const std::vector<cl::sycl::event>& events = {});

template <typename T, typename BatchIndicesT,
          template <typename> class PoolType, typename Backend>
SNNStatus sublaunch(
    typename Backend::template pointer_type<T const> input,
    typename Backend::template pointer_type<T const> rois,
    typename Backend::template pointer_type<BatchIndicesT const> batch_indices,
    typename Backend::template pointer_type<T> output,
    const RoiAlignParams& rap, Backend& backend,
    const std::vector<cl::sycl::event>& events = {}) {
  auto inp_mem = backend.get_mem_object(
      input, rap.batch * rap.channels * rap.in_height * rap.in_width);
  auto rois_mem = backend.get_mem_object(rois, rap.num_rois * rap.roi_cols);
  auto batch_indices_mem = backend.get_mem_object(batch_indices, rap.num_rois);
  auto outp_mem = backend.get_mem_object(
      output, rap.num_rois * rap.channels * rap.out_height * rap.out_width);
  auto queue = backend.get_queue();

  return internal::launch_roi_align<T, BatchIndicesT, PoolType>(
      inp_mem, rois_mem, batch_indices_mem, outp_mem, rap, queue, events);
}

}  // namespace internal
}  // namespace roi_align
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_ROI_ALIGN_LAUNCH_INTERNAL_H_
