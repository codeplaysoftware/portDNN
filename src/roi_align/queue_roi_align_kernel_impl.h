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

#ifndef PORTDNN_SRC_ROI_ALIGN_QUEUE_IMPL_H_
#define PORTDNN_SRC_ROI_ALIGN_QUEUE_IMPL_H_

#include "portdnn/helpers/ratio.h"
#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/roi_align/params.h"

#include "src/roi_align/kernels.h"
#include "src/roi_align/queue_roi_align_kernel.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace roi_align {
namespace internal {

template <typename T, typename BatchIndicesT, typename Index,
          template <typename> class PoolType, template <typename> class MemObj>
SNNStatus queue_roi_align(MemObj<T const>& in_mem, MemObj<T const>& rois_mem,
                          MemObj<BatchIndicesT const>& batch_indices_mem,
                          MemObj<T>& out_mem, RoiAlignParams const& rap,
                          size_t threads, cl::sycl::queue& queue,
                          const std::vector<cl::sycl::event>& events) {
  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.depends_on(events);
    auto input = in_mem.read_mem(cgh);
    auto rois = rois_mem.read_mem(cgh);
    auto batch_indices = batch_indices_mem.read_mem(cgh);
    auto output = out_mem.write_mem(cgh);
    RoiAlignOp<T, BatchIndicesT, Index, PoolType, is_usm_obj_v<MemObj<T>, T>>
        roi_align{input, rois, batch_indices, output, rap, threads};

    cgh.parallel_for(cl::sycl::range<1>{threads}, roi_align);
  });

  return {event, StatusCode::OK};
}

}  // namespace internal
}  // namespace roi_align
}  // namespace sycldnn

#endif  // PORTDNN_SRC_ROI_ALIGN_QUEUE_IMPL_H_
