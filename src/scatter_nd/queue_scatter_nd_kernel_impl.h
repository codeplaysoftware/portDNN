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

#ifndef PORTDNN_SRC_SCATTER_ND_QUEUE_IMPL_H
#define PORTDNN_SRC_SCATTER_ND_QUEUE_IMPL_H

#include "portdnn/status.h"

#include "portdnn/mem_object.h"
#include "portdnn/scatter_nd/operators.h"
#include "portdnn/scatter_nd/sizes.h"
#include "src/scatter_nd/kernels.h"

namespace sycldnn {
namespace scatter_nd {
namespace internal {

template <typename T, typename Index, typename ScatterNDType, int IndexDepth,
          int VectorWidth, template <typename> class MemObj>
SNNStatus queue_scatter_nd(MemObj<Index const>& ind_mem,
                           MemObj<T const>& upd_mem, MemObj<T>& out_mem,
                           const ScatterNDSizes& sizes, cl::sycl::queue& queue,
                           const std::vector<cl::sycl::event>& events) {
  size_t num_updates = sizes.num_updates;
  size_t slice_size = sizes.slice_size;
  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.depends_on(events);
    auto indices_mem = ind_mem.read_mem(cgh);
    auto update_mem = upd_mem.read_mem(cgh);
    auto output_mem = out_mem.write_mem(cgh);
    ScatterNDOp<T, Index, ScatterNDType, IndexDepth, VectorWidth,
                is_usm_obj_v<MemObj<T>, T>>
        op{indices_mem, update_mem, output_mem, sizes};

    cgh.parallel_for(cl::sycl::range<2>{num_updates, slice_size / VectorWidth},
                     op);
  });
  return {event, StatusCode::OK};
}

}  // namespace internal
}  // namespace scatter_nd
}  // namespace sycldnn

#endif  // PORTDNN_SRC_SCATTER_ND_QUEUE_IMPL_H
