
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
#ifndef SYCLDNN_SRC_GATHER_QUEUE_IMPL_H_
#define SYCLDNN_SRC_GATHER_QUEUE_IMPL_H_

#include "src/gather/kernels.h"
#include "src/gather/queue_kernel.h"

namespace sycldnn {
namespace gather {
namespace internal {

template <typename T, typename Index>
SNNStatus queue_gather(BaseMemObject<T const>& in_mem,
                       BaseMemObject<Index const>& indices_mem,
                       BaseMemObject<T>& out_mem, const GatherSizes& gs,
                       cl::sycl::queue& queue) {
  size_t const n_items = gs.output_size;

  Index indices_size = gs.indices_size;
  Index block_size = gs.block_size;
  Index max_index = gs.indices_max;
  Index output_size = gs.output_size;

  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    auto input = in_mem.read_accessor(cgh);
    auto indices = indices_mem.read_accessor(cgh);
    auto output = out_mem.write_accessor(cgh);

    GatherOp<T, Index> gatherFunc{input,      indices,   output,
                                  block_size, max_index, indices_size,
                                  output_size};

    cgh.parallel_for(cl::sycl::range<1>{n_items}, gatherFunc);
  });
  return {event, StatusCode::OK};
}

}  // namespace internal
}  // namespace gather
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_GATHER_QUEUE_IMPL_H_
