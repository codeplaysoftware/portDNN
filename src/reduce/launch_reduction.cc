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
#include "src/reduce/queue_reduction.h"
#include "sycldnn/internal/helpers/types.h"
#include "sycldnn/internal/reduce/launch.h"
#include "sycldnn/mem_object.h"
#include "sycldnn/reduce/operators.h"

namespace sycldnn {
namespace reduce {
namespace internal {

#ifdef SNN_DISABLE_SYCL_PROGRAM
// Launch the reduce kernel for the passed parameters.
template <typename T, typename Op>
SNNStatus launch(BaseMemObject<T const>& input, BaseMemObject<T>& output,
                 int batches, int outer, int inner, cl::sycl::queue& queue) {
  return queue_default_kernel<T, int, Op>(input, output, batches, outer, inner,
                                          outer, queue);
}
#else
// Launch the reduce kernel for the passed parameters.
template <typename T, typename Op>
SNNStatus launch(BaseMemObject<T const>& input, BaseMemObject<T>& output,
                 int batches, int outer, int inner, cl::sycl::queue& queue,
                 cl::sycl::program& program, bool supports_subgroup,
                 sycldnn::internal::types::KernelSubgroupSizesMap&
                     max_kernel_sub_group_sizes) {
#if SNN_ENABLE_SUBGROUPS
  if (supports_subgroup && inner == 1) {
    return queue_subgroup_kernel<T, int, Op>(input, output, batches, outer,
                                             inner, queue, program,
                                             max_kernel_sub_group_sizes);
  }
#endif
  SNN_UNUSED_VAR(program);
  SNN_UNUSED_VAR(supports_subgroup);
  SNN_UNUSED_VAR(max_kernel_sub_group_sizes);
  return queue_default_kernel<T, int, Op>(input, output, batches, outer, inner,
                                          outer, queue);
}
#endif

#ifdef SNN_DISABLE_SYCL_PROGRAM
#define INSTANTIATE_LAUNCHER(DTYPE, OP)                                  \
  template SNN_EXPORT SNNStatus launch<DTYPE, OP>(                       \
      BaseMemObject<DTYPE const> & input, BaseMemObject<DTYPE> & output, \
      int batches, int outer, int inner, cl::sycl::queue& queue);
#else
#define INSTANTIATE_LAUNCHER(DTYPE, OP)                                  \
  template SNN_EXPORT SNNStatus launch<DTYPE, OP>(                       \
      BaseMemObject<DTYPE const> & input, BaseMemObject<DTYPE> & output, \
      int batches, int outer, int inner, cl::sycl::queue& queue,         \
      cl::sycl::program& program, bool supports_subgroup,                \
      sycldnn::internal::types::KernelSubgroupSizesMap&                  \
          max_kernel_sub_group_sizes);
#endif

#define INSTANTIATE_FOR_TYPE(DTYPE) \
  INSTANTIATE_LAUNCHER(DTYPE, Add)  \
  INSTANTIATE_LAUNCHER(DTYPE, Mean) \
  INSTANTIATE_LAUNCHER(DTYPE, Max)  \
  INSTANTIATE_LAUNCHER(DTYPE, Min)

INSTANTIATE_FOR_TYPE(float);

#ifdef SNN_USE_DOUBLE
INSTANTIATE_FOR_TYPE(double);
#endif  // SNN_USE_DOUBLE

#ifdef SNN_USE_HALF
INSTANTIATE_FOR_TYPE(cl::sycl::half);
#endif  // SNN_USE_HALF

#undef INSTANTIATE_FOR_TYPE
#undef INSTANTIATE_LAUNCHER

}  // namespace internal
}  // namespace reduce
}  // namespace sycldnn
