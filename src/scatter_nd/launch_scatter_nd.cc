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

#if defined(SNN_TEST_SYCLBLAS)
#include "portdnn/backend/sycl_blas_backend.h"
#else
#include "portdnn/backend/snn_backend.h"
#endif

#include "portdnn/helpers/dims.h"
#include "portdnn/status.h"

#include "portdnn/scatter_nd/operators.h"
#include "portdnn/scatter_nd/sizes.h"

#include "portdnn/helpers/mem_utils.h"
#include "portdnn/internal/scatter_nd/launch_internal.h"
#include "src/scatter_nd/queue_scatter_nd_kernel.h"

#include <CL/sycl.hpp>

#include "portdnn/export.h"

namespace sycldnn {
namespace scatter_nd {
namespace internal {

/**
 * Helper struct that tries to use cl::sycl::vec for slices greater than 1.
 * Currently, only the Assign operator can support this
 *
 */
template <typename T, typename Index, typename ScatterNDType, int IndexDepth,
          template <typename> class MemObj>
struct VectorWidthLauncher {
  // Default method for operators that aren't the Assign operator.
  static SNNStatus launch_with_vector_width(
      MemObj<Index const>& ind_mem, MemObj<T const>& upd_mem,
      MemObj<T>& out_mem, ScatterNDSizes const& sizes, cl::sycl::queue& queue,
      const std::vector<cl::sycl::event>& events) {
    return queue_scatter_nd<T, Index, ScatterNDType, IndexDepth, 1>(
        ind_mem, upd_mem, out_mem, sizes, queue, events);
  }
};

/**
 * Specialised method for the Assign operator.
 *
 */
template <typename T, typename Index, int IndexDepth,
          template <typename> class MemObj>
struct VectorWidthLauncher<T, Index, Assign, IndexDepth, MemObj> {
  static SNNStatus launch_with_vector_width(
      MemObj<Index const>& ind_mem, MemObj<T const>& upd_mem,
      MemObj<T>& out_mem, ScatterNDSizes const& sizes, cl::sycl::queue& queue,
      const std::vector<cl::sycl::event>& events) {
    if (sizes.slice_size % 4 == 0) {
      return queue_scatter_nd<T, Index, Assign, IndexDepth, 4>(
          ind_mem, upd_mem, out_mem, sizes, queue, events);
    } else if (sizes.slice_size % 2 == 0) {
      return queue_scatter_nd<T, Index, Assign, IndexDepth, 2>(
          ind_mem, upd_mem, out_mem, sizes, queue, events);
    } else {
      return queue_scatter_nd<T, Index, Assign, IndexDepth, 1>(
          ind_mem, upd_mem, out_mem, sizes, queue, events);
    }
  }
};

/**
 * The internal scatter_nd launcher.
 *
 */
template <typename DType, typename IType, typename ScatterNDType,
          int IndexDepth, template <typename> class MemObj>
SNNStatus launch_scatter_nd(MemObj<DType const>& in_mem,
                            MemObj<IType const>& ind_mem,
                            MemObj<DType const>& upd_mem,
                            MemObj<DType>& out_mem, ScatterNDSizes const& sizes,
                            cl::sycl::queue& queue,
                            const std::vector<cl::sycl::event>& events) {
  // Fill output buffer with input data
  auto e = sycldnn::helpers::cpy(in_mem, out_mem, queue, events);

  SNNStatus status =
      VectorWidthLauncher<DType, IType, ScatterNDType, IndexDepth, MemObj>::
          launch_with_vector_width(ind_mem, upd_mem, out_mem, sizes, queue,
                                   std::vector<cl::sycl::event>{e});
  return status;
}

#define INSTANTIATE_LAUNCH(DTYPE, ITYPE, ScatterNDType, ScatterNDIndexDepth, \
                           MEM_OBJ)                                          \
  template SNN_EXPORT SNNStatus launch_scatter_nd<                           \
      DTYPE, ITYPE, ScatterNDType, ScatterNDIndexDepth, MEM_OBJ>(            \
      MEM_OBJ<DTYPE const> & input, MEM_OBJ<ITYPE const> & indices,          \
      MEM_OBJ<DTYPE const> & update, MEM_OBJ<DTYPE> & output,                \
      ScatterNDSizes const& sizes, cl::sycl::queue& queue,                   \
      const std::vector<cl::sycl::event>& events);

#define INSTANTIATE_LAUNCH_FOR_EACH_INDEX_DEPTH(DTYPE, ITYPE, ScatterNDType, \
                                                MEM_OBJ)                     \
  INSTANTIATE_LAUNCH(DTYPE, ITYPE, ScatterNDType, 1, MEM_OBJ);               \
  INSTANTIATE_LAUNCH(DTYPE, ITYPE, ScatterNDType, 2, MEM_OBJ);               \
  INSTANTIATE_LAUNCH(DTYPE, ITYPE, ScatterNDType, 3, MEM_OBJ);               \
  INSTANTIATE_LAUNCH(DTYPE, ITYPE, ScatterNDType, 4, MEM_OBJ);

#define INSTANTIATE_LAUNCH_FOR_TYPE_ASSIGN_OP(DTYPE, MEM_OBJ) \
  INSTANTIATE_LAUNCH_FOR_EACH_INDEX_DEPTH(                    \
      DTYPE, int32_t, sycldnn::scatter_nd::Assign, MEM_OBJ);  \
  INSTANTIATE_LAUNCH_FOR_EACH_INDEX_DEPTH(                    \
      DTYPE, int64_t, sycldnn::scatter_nd::Assign, MEM_OBJ);

#ifdef SNN_ENABLE_USM
INSTANTIATE_LAUNCH_FOR_TYPE_ASSIGN_OP(uint8_t, USMMemObject);
INSTANTIATE_LAUNCH_FOR_TYPE_ASSIGN_OP(uint16_t, USMMemObject);
INSTANTIATE_LAUNCH_FOR_TYPE_ASSIGN_OP(uint32_t, USMMemObject);
INSTANTIATE_LAUNCH_FOR_TYPE_ASSIGN_OP(uint64_t, USMMemObject);
#endif
INSTANTIATE_LAUNCH_FOR_TYPE_ASSIGN_OP(uint8_t, BufferMemObject);
INSTANTIATE_LAUNCH_FOR_TYPE_ASSIGN_OP(uint16_t, BufferMemObject);
INSTANTIATE_LAUNCH_FOR_TYPE_ASSIGN_OP(uint32_t, BufferMemObject);
INSTANTIATE_LAUNCH_FOR_TYPE_ASSIGN_OP(uint64_t, BufferMemObject);

#define INSTANTIATE_LAUNCH_FOR_EACH_OP(DTYPE, ITYPE, MEM_OBJ)                 \
  INSTANTIATE_LAUNCH_FOR_EACH_INDEX_DEPTH(DTYPE, ITYPE,                       \
                                          sycldnn::scatter_nd::Add, MEM_OBJ); \
  INSTANTIATE_LAUNCH_FOR_EACH_INDEX_DEPTH(DTYPE, ITYPE,                       \
                                          sycldnn::scatter_nd::Sub, MEM_OBJ); \
  INSTANTIATE_LAUNCH_FOR_EACH_INDEX_DEPTH(DTYPE, ITYPE,                       \
                                          sycldnn::scatter_nd::Mul, MEM_OBJ); \
  INSTANTIATE_LAUNCH_FOR_EACH_INDEX_DEPTH(DTYPE, ITYPE,                       \
                                          sycldnn::scatter_nd::Div, MEM_OBJ);

#define INSTANTIATE_LAUNCH_FOR_TYPE(DTYPE, MEM_OBJ)        \
  INSTANTIATE_LAUNCH_FOR_EACH_OP(DTYPE, int32_t, MEM_OBJ); \
  INSTANTIATE_LAUNCH_FOR_EACH_OP(DTYPE, int64_t, MEM_OBJ);

#ifdef SNN_ENABLE_USM
INSTANTIATE_LAUNCH_FOR_TYPE(int32_t, USMMemObject)
INSTANTIATE_LAUNCH_FOR_TYPE(int64_t, USMMemObject)
INSTANTIATE_LAUNCH_FOR_TYPE(float, USMMemObject)
#endif
INSTANTIATE_LAUNCH_FOR_TYPE(int32_t, BufferMemObject)
INSTANTIATE_LAUNCH_FOR_TYPE(int64_t, BufferMemObject)
INSTANTIATE_LAUNCH_FOR_TYPE(float, BufferMemObject)
#ifdef SNN_USE_DOUBLE
#ifdef SNN_ENABLE_USM
INSTANTIATE_LAUNCH_FOR_TYPE(double, USMMemObject)
#endif
INSTANTIATE_LAUNCH_FOR_TYPE(double, BufferMemObject)
#endif
#ifdef SNN_USE_HALF
#ifdef SNN_ENABLE_USM
INSTANTIATE_LAUNCH_FOR_TYPE(cl::sycl::half, USMMemObject)
#endif
INSTANTIATE_LAUNCH_FOR_TYPE(cl::sycl::half, BufferMemObject)
#endif

#undef INSTANTIATE_LAUNCH
#undef INSTANTIATE_LAUNCH_FOR_EACH_RANK
#undef INSTANTIATE_LAUNCH_FOR_EACH_OP
#undef INSTANTIATE_LAUNCH_FOR_TYPE
#undef INSTANTIATE_LAUNCH_FOR_TYPE_ASSIGN_OP

}  // namespace internal
}  // namespace scatter_nd
}  // namespace sycldnn
