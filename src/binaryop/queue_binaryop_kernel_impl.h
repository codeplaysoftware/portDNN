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

#ifndef PORTDNN_SRC_BINARYOP_QUEUE_IMPL_H_
#define PORTDNN_SRC_BINARYOP_QUEUE_IMPL_H_

#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/helpers/dims.h"
#include "src/binaryop/queue_binaryop_kernel.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace binaryop {
namespace internal {

template <typename Kernel, typename T, typename Index,
          template <typename> class MemObj>
SNNStatus queue_binaryop(MemObj<T const>& lhs, MemObj<T const>& rhs,
                         MemObj<T>& out, const std::vector<Index>& lhs_dims,
                         const std::vector<Index>& rhs_dims,
                         const std::vector<Index>& out_dims,
                         cl::sycl::queue& queue,
                         const std::vector<cl::sycl::event>& events) {
  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.depends_on(events);
    auto lhs_mem = lhs.read_mem(cgh);
    auto rhs_mem = rhs.read_mem(cgh);
    auto out_mem = out.write_mem(cgh);
    Kernel binary_op(lhs_mem, rhs_mem, out_mem, lhs_dims, rhs_dims, out_dims);
    cgh.parallel_for(binary_op.get_range(), binary_op);
  });

  return {event, StatusCode::OK};
}

}  // namespace internal
}  // namespace binaryop
}  // namespace sycldnn

#endif  // PORTDNN_SRC_BINARYOP_QUEUE_IMPL_H_
