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

#ifndef SYCLDNN_SRC_BINARYOP_QUEUE_IMPL_H_
#define SYCLDNN_SRC_BINARYOP_QUEUE_IMPL_H_

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "src/binaryop/kernels.h"
#include "src/binaryop/queue_binaryop_kernel.h"

#include <CL/sycl.hpp>

namespace sycldnn {
namespace binaryop {
namespace internal {

template <typename T, typename Op, typename Index, int VectorWidth>
SNNStatus queue_binaryop(BaseMemObject<T const>& lhs,
                         BaseMemObject<T const>& rhs, BaseMemObject<T>& output,
                         const BinaryParams& params, cl::sycl::queue& queue) {
  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    auto in1 = lhs.read_accessor(cgh);
    auto in2 = rhs.read_accessor(cgh);
    auto out = output.write_accessor(cgh);
    auto x_range = static_cast<size_t>(params.lhs_items / params.rhs_items);
    auto y_range = static_cast<size_t>(params.rhs_items);
    BinaryOp<T, Op, Index, VectorWidth> binary_op{in1, in2, out};

    cgh.parallel_for(cl::sycl::range<2>{x_range, y_range / VectorWidth},
                     binary_op);
  });

  return {event, StatusCode::OK};
}

}  // namespace internal
}  // namespace binaryop
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_BINARYOP_QUEUE_IMPL_H_
