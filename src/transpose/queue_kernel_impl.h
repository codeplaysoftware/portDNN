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
#ifndef PORTDNN_SRC_TRANSPOSE_QUEUE_KERNEL_IMPL_H_
#define PORTDNN_SRC_TRANSPOSE_QUEUE_KERNEL_IMPL_H_

#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "src/transpose/kernels.h"

namespace sycldnn {
namespace transpose {
namespace internal {

template <typename T, typename Index, int ND, template <typename> class MemObj>
SNNStatus queue_kernel(MemObj<T const>& input_mem, MemObj<T>& output_mem,
                       std::vector<int> const& dimensions,
                       std::vector<int> const& permutation,
                       cl::sycl::queue& queue,
                       const std::vector<cl::sycl::event>& events) {
  constexpr bool is_usm = is_usm_obj_v<MemObj<T>, T>;
  using Functor = TransposeKernel<T, Index, ND, is_usm>;
  auto event = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.depends_on(events);
    auto input = input_mem.read_mem(cgh);
    auto output = output_mem.write_mem(cgh);

    size_t const n_threads = std::accumulate(
        begin(dimensions), end(dimensions), static_cast<size_t>(1),
        [](size_t a, int b) { return a * b; });

    Functor functor{input, output, dimensions, permutation};

    cgh.parallel_for(cl::sycl::range<1>{n_threads}, functor);
  });
  return {event, StatusCode::OK};
}

}  // namespace internal
}  // namespace transpose
}  // namespace sycldnn

#endif  // PORTDNN_SRC_TRANSPOSE_QUEUE_KERNEL_IMPL_H_
