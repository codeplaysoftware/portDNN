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

#include "portdnn/internal/gather/launch.h"
#include "src/gather/queue_kernel.h"

namespace sycldnn {
namespace gather {
namespace internal {

template <typename T, typename Index, template <typename> class MemObj>
SNNStatus launch_impl(MemObj<T const>& input, MemObj<Index const>& indices,
                      MemObj<T>& output, const GatherSizes& gs,
                      cl::sycl::queue& queue,
                      const std::vector<cl::sycl::event>& events) {
  return queue_gather<T, Index>(input, indices, output, gs, queue, events);
}

#define INSTANTIATE_FOR_TYPE(DTYPE, MEM_OBJ)                                  \
  template SNN_EXPORT SNNStatus launch_impl<DTYPE, int32_t>(                  \
      MEM_OBJ<DTYPE const> & input, MEM_OBJ<int32_t const> & indices,         \
      MEM_OBJ<DTYPE> & output, const GatherSizes& gs, cl::sycl::queue& queue, \
      const std::vector<cl::sycl::event>& events);                            \
  template SNN_EXPORT SNNStatus launch_impl<DTYPE, int64_t>(                  \
      MEM_OBJ<DTYPE const> & input, MEM_OBJ<int64_t const> & indices,         \
      MEM_OBJ<DTYPE> & output, const GatherSizes& gs, cl::sycl::queue& queue, \
      const std::vector<cl::sycl::event>& events)

#ifdef SNN_ENABLE_USM
INSTANTIATE_FOR_TYPE(uint8_t, USMMemObject);
INSTANTIATE_FOR_TYPE(uint16_t, USMMemObject);
INSTANTIATE_FOR_TYPE(uint32_t, USMMemObject);
INSTANTIATE_FOR_TYPE(uint64_t, USMMemObject);
#endif

INSTANTIATE_FOR_TYPE(uint8_t, BufferMemObject);
INSTANTIATE_FOR_TYPE(uint16_t, BufferMemObject);
INSTANTIATE_FOR_TYPE(uint32_t, BufferMemObject);
INSTANTIATE_FOR_TYPE(uint64_t, BufferMemObject);

#undef INSTANTIATE_FOR_TYPE

}  // namespace internal
}  // namespace gather
}  // namespace sycldnn
