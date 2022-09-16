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

#include "sycldnn/internal/gather/launch.h"
#include "src/gather/queue_kernel.h"

namespace sycldnn {
namespace gather {
namespace internal {

template <typename T, typename Index>
SNNStatus launch_impl(BaseMemObject<T const>& input,
                      BaseMemObject<Index const>& indices,
                      BaseMemObject<T>& output, const GatherSizes& gs,
                      cl::sycl::queue& queue) {
  return queue_gather<T, Index>(input, indices, output, gs, queue);
}

#define INSTANTIATE_FOR_TYPE(DTYPE)                                          \
  template SNN_EXPORT SNNStatus launch_impl<DTYPE, int32_t>(                 \
      BaseMemObject<DTYPE const> & input,                                    \
      BaseMemObject<int32_t const> & indices, BaseMemObject<DTYPE> & output, \
      const GatherSizes& gs, cl::sycl::queue& queue);                        \
  template SNN_EXPORT SNNStatus launch_impl<DTYPE, int64_t>(                 \
      BaseMemObject<DTYPE const> & input,                                    \
      BaseMemObject<int64_t const> & indices, BaseMemObject<DTYPE> & output, \
      const GatherSizes& gs, cl::sycl::queue& queue)

INSTANTIATE_FOR_TYPE(uint8_t);
INSTANTIATE_FOR_TYPE(uint16_t);
INSTANTIATE_FOR_TYPE(uint32_t);
INSTANTIATE_FOR_TYPE(uint64_t);

#undef INSTANTIATE_FOR_TYPE

}  // namespace internal
}  // namespace gather
}  // namespace sycldnn
