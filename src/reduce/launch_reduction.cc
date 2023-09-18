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
#include "portdnn/internal/helpers/types.h"
#include "portdnn/internal/reduce/launch.h"
#include "portdnn/mem_object.h"
#include "portdnn/reduce/operators.h"
#include "src/reduce/queue_reduction.h"

namespace sycldnn {
namespace reduce {
namespace internal {

#ifdef SNN_DISABLE_SYCL_PROGRAM
// Launch the reduce kernel for the passed parameters.
template <typename T, typename Op, template <typename> class MemObj>
SNNStatus launch(MemObj<T const>& input, MemObj<T>& output, int batches,
                 int outer, int inner, cl::sycl::queue& queue,
                 const std::vector<cl::sycl::event>& events) {
  return queue_default_kernel<T, int, Op>(input, output, batches, outer, inner,
                                          outer, queue, events);
}
#else
// Launch the reduce kernel for the passed parameters.
template <typename T, typename Op, template <typename> class MemObj>
SNNStatus launch(MemObj<T const>& input, MemObj<T>& output, int batches,
                 int outer, int inner, cl::sycl::queue& queue,
                 cl::sycl::program& program, bool supports_subgroup,
                 sycldnn::internal::types::KernelSubgroupSizesMap&
                     max_kernel_sub_group_sizes,
                 const std::vector<cl::sycl::event>& events) {
#if SNN_ENABLE_SUBGROUPS
  if (supports_subgroup && inner == 1) {
    return queue_subgroup_kernel<T, int, Op>(
        input, output, batches, outer, inner, queue, program,
        max_kernel_sub_group_sizes, events);
  }
#endif
  SNN_UNUSED_VAR(program);
  SNN_UNUSED_VAR(supports_subgroup);
  SNN_UNUSED_VAR(max_kernel_sub_group_sizes);
  return queue_default_kernel<T, int, Op>(input, output, batches, outer, inner,
                                          outer, queue, events);
}
#endif

#ifdef SNN_DISABLE_SYCL_PROGRAM
#define INSTANTIATE_LAUNCHER(DTYPE, OP, MEMOBJ)                         \
  template SNN_EXPORT SNNStatus launch<DTYPE, OP>(                      \
      MEMOBJ<DTYPE const> & input, MEMOBJ<DTYPE> & output, int batches, \
      int outer, int inner, cl::sycl::queue& queue,                     \
      const std::vector<cl::sycl::event>& events);
#else
#define INSTANTIATE_LAUNCHER(DTYPE, OP, MEMOBJ)                         \
  template SNN_EXPORT SNNStatus launch<DTYPE, OP>(                      \
      MEMOBJ<DTYPE const> & input, MEMOBJ<DTYPE> & output, int batches, \
      int outer, int inner, cl::sycl::queue& queue,                     \
      cl::sycl::program& program, bool supports_subgroup,               \
      sycldnn::internal::types::KernelSubgroupSizesMap&                 \
          max_kernel_sub_group_sizes,                                   \
      const std::vector<cl::sycl::event>& events);
#endif

#define INSTANTIATE_FOR_TYPE(DTYPE, MEMOBJ) \
  INSTANTIATE_LAUNCHER(DTYPE, Add, MEMOBJ)  \
  INSTANTIATE_LAUNCHER(DTYPE, Mean, MEMOBJ) \
  INSTANTIATE_LAUNCHER(DTYPE, Max, MEMOBJ)  \
  INSTANTIATE_LAUNCHER(DTYPE, Min, MEMOBJ)

#ifdef SNN_ENABLE_USM
INSTANTIATE_FOR_TYPE(float, USMMemObject);
#endif
INSTANTIATE_FOR_TYPE(float, BufferMemObject);

#ifdef SNN_USE_DOUBLE
#ifdef SNN_ENABLE_USM
INSTANTIATE_FOR_TYPE(double, USMMemObject);
#endif
INSTANTIATE_FOR_TYPE(double, BufferMemObject);
#endif  // SNN_USE_DOUBLE

#ifdef SNN_USE_HALF
#ifdef SNN_ENABLE_USM
INSTANTIATE_FOR_TYPE(cl::sycl::half, USMMemObject);
#endif
INSTANTIATE_FOR_TYPE(cl::sycl::half, BufferMemObject);
#endif  // SNN_USE_HALF

#undef INSTANTIATE_FOR_TYPE
#undef INSTANTIATE_LAUNCHER

}  // namespace internal
}  // namespace reduce
}  // namespace sycldnn
