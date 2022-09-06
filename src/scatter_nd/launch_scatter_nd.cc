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
#include "sycldnn/backend/sycl_blas_backend.h"
#else
#include "sycldnn/backend/snn_backend.h"
#endif

#include "sycldnn/helpers/dims.h"
#include "sycldnn/status.h"

#include "sycldnn/scatter_nd/operators.h"
#include "sycldnn/scatter_nd/sizes.h"

#include "src/scatter_nd/queue_scatter_nd_kernel.h"
#include "sycldnn/internal/scatter_nd/launch_internal.h"

#include <CL/sycl.hpp>

#include "sycldnn/export.h"

namespace sycldnn {
namespace scatter_nd {
namespace internal {

/**
 * Helper struct that tries to use cl::sycl::vec for slices greater than 1.
 * Currently, only the Assign operator can support this
 *
 */
template <typename T, typename Index, typename ScatterNDType, int IndexDepth>
struct VectorWidthLauncher {
  // Default method for operators that aren't the Assign operator.
  static SNNStatus launch_with_vector_width(BaseMemObject<Index const>& ind_mem,
                                            BaseMemObject<T const>& upd_mem,
                                            BaseMemObject<T>& out_mem,
                                            ScatterNDSizes const& sizes,
                                            cl::sycl::queue& queue) {
    return queue_scatter_nd<T, Index, ScatterNDType, IndexDepth, 1>(
        ind_mem, upd_mem, out_mem, sizes, queue);
  }
};

/**
 * Specialised method for the Assign operator.
 *
 */
template <typename T, typename Index, int IndexDepth>
struct VectorWidthLauncher<T, Index, Assign, IndexDepth> {
  static SNNStatus launch_with_vector_width(BaseMemObject<Index const>& ind_mem,
                                            BaseMemObject<T const>& upd_mem,
                                            BaseMemObject<T>& out_mem,
                                            ScatterNDSizes const& sizes,
                                            cl::sycl::queue& queue) {
    if (sizes.slice_size % 4 == 0) {
      return queue_scatter_nd<T, Index, Assign, IndexDepth, 4>(
          ind_mem, upd_mem, out_mem, sizes, queue);
    } else if (sizes.slice_size % 2 == 0) {
      return queue_scatter_nd<T, Index, Assign, IndexDepth, 2>(
          ind_mem, upd_mem, out_mem, sizes, queue);
    } else {
      return queue_scatter_nd<T, Index, Assign, IndexDepth, 1>(
          ind_mem, upd_mem, out_mem, sizes, queue);
    }
  }
};

/**
 * The internal scatter_nd launcher.
 *
 */
template <typename DType, typename IType, typename ScatterNDType,
          int IndexDepth>
SNNStatus launch_scatter_nd(BaseMemObject<DType const>& in_mem,
                            BaseMemObject<IType const>& ind_mem,
                            BaseMemObject<DType const>& upd_mem,
                            BaseMemObject<DType>& out_mem,
                            ScatterNDSizes const& sizes,
                            cl::sycl::queue& queue) {
  // Fill output buffer with input data
  queue.submit([&](cl::sycl::handler& cgh) {
    auto in_acc = in_mem.read_accessor(cgh).get_accessor();
    auto out_acc = out_mem.write_accessor(cgh).get_accessor();
    cgh.copy(in_acc, out_acc);
  });

  SNNStatus status =
      VectorWidthLauncher<DType, IType, ScatterNDType,
                          IndexDepth>::launch_with_vector_width(ind_mem,
                                                                upd_mem,
                                                                out_mem, sizes,
                                                                queue);
  return status;
}

#define INSTANTIATE_LAUNCH(DTYPE, ITYPE, ScatterNDType, ScatterNDIndexDepth) \
  template SNN_EXPORT SNNStatus                                              \
  launch_scatter_nd<DTYPE, ITYPE, ScatterNDType, ScatterNDIndexDepth>(       \
      BaseMemObject<DTYPE const> & input,                                    \
      BaseMemObject<ITYPE const> & indices,                                  \
      BaseMemObject<DTYPE const> & update, BaseMemObject<DTYPE> & output,    \
      ScatterNDSizes const& sizes, cl::sycl::queue& queue);

#define INSTANTIATE_LAUNCH_FOR_EACH_INDEX_DEPTH(DTYPE, ITYPE, ScatterNDType) \
  INSTANTIATE_LAUNCH(DTYPE, ITYPE, ScatterNDType, 1);                        \
  INSTANTIATE_LAUNCH(DTYPE, ITYPE, ScatterNDType, 2);                        \
  INSTANTIATE_LAUNCH(DTYPE, ITYPE, ScatterNDType, 3);                        \
  INSTANTIATE_LAUNCH(DTYPE, ITYPE, ScatterNDType, 4);

#define INSTANTIATE_LAUNCH_FOR_TYPE_ASSIGN_OP(DTYPE)                    \
  INSTANTIATE_LAUNCH_FOR_EACH_INDEX_DEPTH(DTYPE, int32_t,               \
                                          sycldnn::scatter_nd::Assign); \
  INSTANTIATE_LAUNCH_FOR_EACH_INDEX_DEPTH(DTYPE, int64_t,               \
                                          sycldnn::scatter_nd::Assign);

INSTANTIATE_LAUNCH_FOR_TYPE_ASSIGN_OP(uint8_t);
INSTANTIATE_LAUNCH_FOR_TYPE_ASSIGN_OP(uint16_t);
INSTANTIATE_LAUNCH_FOR_TYPE_ASSIGN_OP(uint32_t);
INSTANTIATE_LAUNCH_FOR_TYPE_ASSIGN_OP(uint64_t);

#define INSTANTIATE_LAUNCH_FOR_EACH_OP(DTYPE, ITYPE)                 \
  INSTANTIATE_LAUNCH_FOR_EACH_INDEX_DEPTH(DTYPE, ITYPE,              \
                                          sycldnn::scatter_nd::Add); \
  INSTANTIATE_LAUNCH_FOR_EACH_INDEX_DEPTH(DTYPE, ITYPE,              \
                                          sycldnn::scatter_nd::Sub); \
  INSTANTIATE_LAUNCH_FOR_EACH_INDEX_DEPTH(DTYPE, ITYPE,              \
                                          sycldnn::scatter_nd::Mul); \
  INSTANTIATE_LAUNCH_FOR_EACH_INDEX_DEPTH(DTYPE, ITYPE,              \
                                          sycldnn::scatter_nd::Div);

#define INSTANTIATE_LAUNCH_FOR_TYPE(DTYPE)        \
  INSTANTIATE_LAUNCH_FOR_EACH_OP(DTYPE, int32_t); \
  INSTANTIATE_LAUNCH_FOR_EACH_OP(DTYPE, int64_t);

INSTANTIATE_LAUNCH_FOR_TYPE(int32_t)
INSTANTIATE_LAUNCH_FOR_TYPE(int64_t)
INSTANTIATE_LAUNCH_FOR_TYPE(float)
#ifdef SNN_USE_DOUBLE
INSTANTIATE_LAUNCH_FOR_TYPE(double)
#endif
#ifdef SNN_USE_HALF
INSTANTIATE_LAUNCH_FOR_TYPE(cl::sycl::half)
#endif

#undef INSTANTIATE_LAUNCH
#undef INSTANTIATE_LAUNCH_FOR_EACH_RANK
#undef INSTANTIATE_LAUNCH_FOR_EACH_OP
#undef INSTANTIATE_LAUNCH_FOR_TYPE
#undef INSTANTIATE_LAUNCH_FOR_TYPE_ASSIGN_OP

}  // namespace internal
}  // namespace scatter_nd
}  // namespace sycldnn