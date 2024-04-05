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
#include "sycldnn/internal/reduce/launch.h"
#include "sycldnn/mem_object.h"
#include "sycldnn/reduce/operators.h"

#include "src/reduce/queue_reduction.h"

namespace sycldnn {
namespace reduce {
namespace internal {

// Launch the reduce kernel for the passed parameters.
template <typename T, typename Op, template <typename> class MemObj>
SNNStatus launch(MemObj<T const>& input, MemObj<T>& output, int batches,
                 int outer, int inner, cl::sycl::queue& queue,
                 const std::vector<cl::sycl::event>& events) {
  return queue_default_kernel<T, int, Op>(input, output, batches, outer, inner,
                                          outer, queue, events);
}

// Launch the reduce kernel for the passed parameters.
template <typename T, typename Op, template <typename> class MemObj>
SNNStatus launch(MemObj<T const>& input_mem, MemObj<T>& output_mem,
                 internal::ReduceParams const& params, cl::sycl::queue& queue,
                 const std::vector<cl::sycl::event>& events) {
  using Index = int;
  auto constexpr local_row_wg = 256;
  auto constexpr local_col_wg = 8;
#ifdef DEFAULT_CPU
  auto constexpr NumSGReductions = 0;
#else
  auto constexpr NumSGReductions = 4;
#endif
  if (params.stride_size == 1) {
    return queue_lm_kernel<T, Index, Op, NumSGReductions, false, false>(
        input_mem, output_mem, params, local_row_wg, local_col_wg, queue,
        events);
  } else if (params.stride_size > 4) {
    return queue_lm_kernel<T, Index, Op, NumSGReductions, true, true>(
        input_mem, output_mem, params, local_row_wg, local_col_wg, queue,
        events);
  } else {
    return queue_lm_kernel<T, Index, Op, NumSGReductions, false, true>(
        input_mem, output_mem, params, local_row_wg, local_col_wg, queue,
        events);
  }
}

#define INSTANTIATE_LAUNCHER(DTYPE, OP, MEMOBJ)                         \
  template SNN_EXPORT SNNStatus launch<DTYPE, OP>(                      \
      MEMOBJ<DTYPE const> & input, MEMOBJ<DTYPE> & output, int batches, \
      int outer, int inner, cl::sycl::queue& queue,                     \
      const std::vector<cl::sycl::event>& events);                      \
                                                                        \
  template SNN_EXPORT SNNStatus launch<DTYPE, OP>(                      \
      MEMOBJ<DTYPE const> & input, MEMOBJ<DTYPE> & output,              \
      internal::ReduceParams const& params, cl::sycl::queue& queue,     \
      const std::vector<cl::sycl::event>& events);

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
