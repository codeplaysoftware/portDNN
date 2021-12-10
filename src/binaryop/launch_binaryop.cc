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

#include "sycldnn/binaryop/operators.h"

#include "sycldnn/internal/binaryop/launch.h"

#include "src/binaryop/queue_binaryop_kernel.h"

#include <CL/sycl.hpp>

#include "sycldnn/export.h"

namespace sycldnn {
namespace binaryop {
namespace internal {

template <typename T, typename Op>
SNNStatus launch_binaryop(BaseMemObject<T const>& lhs,
                          BaseMemObject<T const>& rhs, BaseMemObject<T>& output,
                          int32_t const n_items, cl::sycl::queue& queue) {
  if (n_items % 4 == 0) {
    return queue_binaryop<T, Op, int32_t, 4>(lhs, rhs, output, n_items, queue);
  } else if (n_items % 2 == 0) {
    return queue_binaryop<T, Op, int32_t, 2>(lhs, rhs, output, n_items, queue);
  } else {
    return queue_binaryop<T, Op, int32_t, 1>(lhs, rhs, output, n_items, queue);
  }
}

#define INSTANTIATE_BINARYOP_LAUNCH(DTYPE, OP)                   \
  template SNN_EXPORT SNNStatus launch_binaryop<DTYPE, OP>(      \
      BaseMemObject<DTYPE const> & inp1_access,                  \
      BaseMemObject<DTYPE const> & inp2_access,                  \
      BaseMemObject<DTYPE> & outp_access, int32_t const n_items, \
      cl::sycl::queue& queue)

#define INSTANTIATE_BINARYOP_FOR_TYPE(DTYPE) \
  INSTANTIATE_BINARYOP_LAUNCH(DTYPE, Add);   \
  INSTANTIATE_BINARYOP_LAUNCH(DTYPE, Sub);   \
  INSTANTIATE_BINARYOP_LAUNCH(DTYPE, Mul);   \
  INSTANTIATE_BINARYOP_LAUNCH(DTYPE, Div);

INSTANTIATE_BINARYOP_FOR_TYPE(float);

#ifdef SNN_USE_HALF
INSTANTIATE_BINARYOP_FOR_TYPE(cl::sycl::half);
#endif  // SNN_USE_HALF

#ifdef SNN_USE_DOUBLE
INSTANTIATE_BINARYOP_FOR_TYPE(double);
#endif  // SNN_USE_DOUBLE

}  // namespace internal
}  // namespace binaryop
}  // namespace sycldnn
