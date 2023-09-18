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
#include "portdnn/internal/matmul/launch.h"
#include "portdnn/matmul/params.h"

#include "portdnn/mem_object.h"

#include "src/matmul/queue_kernel.h"

namespace sycldnn {
namespace matmul {
namespace internal {
namespace {

// Launch the kernel specified by the template parameters.
template <typename T, bool TransposeLHS, bool TransposeRHS, int RowTile,
          int AccTile, int ColTile, template <typename> class MemObj>
SNNStatus launch_with_tiles(MemObj<T const>& lhs, MemObj<T const>& rhs,
                            MemObj<T>& output, MatmulParams const& params,
                            cl::sycl::queue& queue, size_t wg_rows,
                            size_t wg_cols, size_t wg_batch,
                            const std::vector<cl::sycl::event>& events) {
  auto kernel = ((params.m % RowTile == 0) && (params.k % AccTile == 0) &&
                 (params.n % ColTile == 0))
                    ? queue_kernel<T, int, TransposeLHS, TransposeRHS, RowTile,
                                   AccTile, ColTile, false, MemObj>
                    : queue_kernel<T, int, TransposeLHS, TransposeRHS, RowTile,
                                   AccTile, ColTile, true, MemObj>;
  return kernel(lhs, rhs, output, params, queue, wg_rows, wg_cols, wg_batch,
                events);
}

}  // namespace

// Launch the matrix multiply kernel for the passed parameters.
template <typename T, bool TransposeLHS, bool TransposeRHS,
          template <typename> class MemObj>
SNNStatus launch(MemObj<T const>& lhs, MemObj<T const>& rhs, MemObj<T>& output,
                 MatmulParams const& params, cl::sycl::queue& queue,
                 const std::vector<cl::sycl::event>& events) {
  return launch_with_tiles<T, TransposeLHS, TransposeRHS, 4, 4, 4, MemObj>(
      lhs, rhs, output, params, queue, 8, 4, 1, events);
}

#define INSTANTIATE_LAUNCHER(DTYPE, TLHS, TRHS, MEMOBJ)            \
  template SNN_EXPORT SNNStatus launch<DTYPE, TLHS, TRHS, MEMOBJ>( \
      MEMOBJ<DTYPE const> & input, MEMOBJ<DTYPE const> & filter,   \
      MEMOBJ<DTYPE> & output, MatmulParams const& params,          \
      cl::sycl::queue& queue, const std::vector<cl::sycl::event>& events);

#ifdef SNN_ENABLE_USM
#define INSTANTIATE_FOR_MEMOBJ(DTYPE, TLHS, TRHS)          \
  INSTANTIATE_LAUNCHER(DTYPE, TLHS, TRHS, BufferMemObject) \
  INSTANTIATE_LAUNCHER(DTYPE, TLHS, TRHS, USMMemObject)
#else
#define INSTANTIATE_FOR_MEMOBJ(DTYPE, TLHS, TRHS) \
  INSTANTIATE_LAUNCHER(DTYPE, TLHS, TRHS, BufferMemObject)
#endif  // SNN_ENABLE_USM

#define INSTANTIATE_FOR_TYPE(DTYPE)          \
  INSTANTIATE_FOR_MEMOBJ(DTYPE, true, true)  \
  INSTANTIATE_FOR_MEMOBJ(DTYPE, false, true) \
  INSTANTIATE_FOR_MEMOBJ(DTYPE, true, false) \
  INSTANTIATE_FOR_MEMOBJ(DTYPE, false, false)

INSTANTIATE_FOR_TYPE(float);

#ifdef SNN_USE_DOUBLE
INSTANTIATE_FOR_TYPE(double);
#endif  // SNN_USE_DOUBLE

#ifdef SNN_USE_HALF
INSTANTIATE_FOR_TYPE(cl::sycl::half);
#endif  // SNN_USE_HALF

#undef INSTANTIATE_FOR_TYPE
#undef INSTANTIATE_FOR_MEMOBJ
#undef INSTANTIATE_LAUNCHER

}  // namespace internal
}  // namespace matmul
}  // namespace sycldnn
