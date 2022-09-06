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

#ifndef SYCLDNN_SRC_SCATTER_ND_QUEUE_IMPL_H
#define SYCLDNN_SRC_SCATTER_ND_QUEUE_IMPL_H

#include "sycldnn/status.h"

#include "src/scatter_nd/kernels.h"
#include "sycldnn/mem_object.h"
#include "sycldnn/scatter_nd/operators.h"
#include "sycldnn/scatter_nd/sizes.h"

namespace sycldnn {
namespace scatter_nd {
namespace internal {

template <typename T, typename Index, typename ScatterNDType, int IndexDepth,
          int VectorWidth>
SNNStatus queue_scatter_nd(BaseMemObject<Index const>& ind_mem,
                           BaseMemObject<T const>& upd_mem,
                           BaseMemObject<T>& out_mem,
                           const ScatterNDSizes& sizes,
                           cl::sycl::queue& queue) {
  size_t num_updates = sizes.num_updates;
  size_t slice_size = sizes.slice_size;
  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    auto indices_acc = ind_mem.read_accessor(cgh);
    auto update_acc = upd_mem.read_accessor(cgh);
    auto output_acc = out_mem.write_accessor(cgh);
    ScatterNDOp<T, Index, ScatterNDType, IndexDepth, VectorWidth> op{
        indices_acc, update_acc, output_acc, sizes};

    cgh.parallel_for(cl::sycl::range<2>{num_updates, slice_size / VectorWidth},
                     op);
  });
  return {event, StatusCode::OK};
}

}  // namespace internal
}  // namespace scatter_nd
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_SCATTER_ND_QUEUE_IMPL_H
