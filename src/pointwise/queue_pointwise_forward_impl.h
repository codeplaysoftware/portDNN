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

#ifndef PORTDNN_SRC_POINTWISE_QUEUE_POINTWISE_FORWARD_IMPL_H_
#define PORTDNN_SRC_POINTWISE_QUEUE_POINTWISE_FORWARD_IMPL_H_

#include "portdnn/helpers/ratio.h"
#include "portdnn/mem_object.h"
#include "portdnn/pointwise/direction.h"
#include "portdnn/pointwise/operators.h"
#include "portdnn/status.h"

#include "src/pointwise/kernels.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace pointwise {
namespace internal {

/**
 * Submits a pointwise operation to the queue with number of threads equal to
 * the output size scaled by the vector size.
 */
template <typename T, typename Index, template <typename> class PointwiseType,
          typename Direction, int VectorWidth, template <typename> class MemObj>
SNNStatus queue_pointwise(MemObj<T const>& in_mem, MemObj<T>& out_mem,
                          Index const n_items, cl::sycl::queue& queue,
                          const std::vector<cl::sycl::event>& events) {
  constexpr bool is_usm = is_usm_obj_v<MemObj<T>, T>;
  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.depends_on(events);
    auto input = in_mem.read_mem(cgh);
    auto output = out_mem.write_mem(cgh);
    Index const n_vecs = n_items / VectorWidth;
    size_t const n_threads = helpers::round_up_to_nearest_multiple(n_vecs, 64);
    PointwiseOp<T, Index, PointwiseType, Direction, VectorWidth, is_usm>
        pointwise_op{input, output, n_vecs};
    cgh.parallel_for(cl::sycl::range<1>{n_threads}, pointwise_op);
  });

  return {event, StatusCode::OK};
}

}  // namespace internal
}  // namespace pointwise
}  // namespace sycldnn

#endif  // PORTDNN_SRC_POINTWISE_QUEUE_POINTWISE_FORWARD_IMPL_H_
