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

#ifndef SYCLDNN_SRC_BINARYOP_QUEUE_H_
#define SYCLDNN_SRC_BINARYOP_QUEUE_H_

#include <CL/sycl.hpp>
#include <vector>

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

namespace sycldnn {
namespace binaryop {
namespace internal {

template <typename Kernel, typename T, typename Index>
SNNStatus queue_binaryop(BaseMemObject<T const>& lhs,
                         BaseMemObject<T const>& rhs, BaseMemObject<T>& out,
                         const std::vector<Index>& lhs_dims,
                         const std::vector<Index>& rhs_dims,
                         const std::vector<Index>& out_dims,
                         cl::sycl::queue& queue);

}  // namespace internal
}  // namespace binaryop
}  // namespace sycldnn

#endif  // SYCLDNN_SRC_BINARYOP_QUEUE_H_
