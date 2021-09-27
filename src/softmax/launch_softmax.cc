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

#include "sycldnn/status.h"

#include "sycldnn/softmax/operators.h"
#include "sycldnn/softmax/params.h"

#include "src/softmax/queue_softmax_kernel.h"
#include "sycldnn/internal/softmax/launch_internal.h"

#include <CL/sycl.hpp>

#include "sycldnn/export.h"

namespace sycldnn {
namespace softmax {
namespace internal {

inline bool can_use_vector_width(SoftmaxParams const& params, int w) {
  return params.channels % w == 0;
};

template <typename T, typename Index, typename SoftmaxType, typename Backend>
SNNStatus launch_with_index(
    typename Backend::template pointer_type<T const>& input,
    typename Backend::template pointer_type<T>& workspace,
    typename Backend::template pointer_type<T>& output,
    SoftmaxParams const& params, Backend& backend) {
  if (can_use_vector_width(params, 4)) {
    return queue_softmax<T, Index, SoftmaxType, Backend, 4>(
        input, workspace, output, params, backend);
  } else if (can_use_vector_width(params, 2)) {
    return queue_softmax<T, Index, SoftmaxType, Backend, 2>(
        input, workspace, output, params, backend);
  } else {
    return queue_softmax<T, Index, SoftmaxType, Backend, 1>(
        input, workspace, output, params, backend);
  }
}
/**
 * The internal softmax launcher.
 *
 * Performs an element-wise exponentiation, followed by reduction
 * and then the pointwise division.
 */
template <typename T, typename SoftmaxType, typename Backend>
SNNStatus launch_softmax_forward(
    typename Backend::template pointer_type<T const>& input,
    typename Backend::template pointer_type<T>& workspace,
    typename Backend::template pointer_type<T>& output,
    SoftmaxParams const& params, Backend& backend) {
  auto total_size = params.batch * params.rows * params.cols * params.channels;
  if (total_size > std::numeric_limits<int64_t>::max()) {
    SNNStatus index_too_large;
    index_too_large.status = StatusCode::IndexExceeded;
    return index_too_large;
  } else if (total_size > std::numeric_limits<int32_t>::max()) {
#ifdef SNN_USE_INT64
    return launch_with_index<T, int64_t, SoftmaxType>(input, workspace, output,
                                                      params, backend);
#else
    SNNStatus index_too_large;
    index_too_large.status = StatusCode::IndexExceeded;
    return index_too_large;
#endif  // SNN_USE_INT64
  } else {
    return launch_with_index<T, int32_t, SoftmaxType>(input, workspace, output,
                                                      params, backend);
  }
}

#define INSTANTIATE_LAUNCH(DTYPE, SoftMaxType, Backend)             \
  template SNN_EXPORT SNNStatus                                     \
  launch_softmax_forward<DTYPE, SoftMaxType, Backend>(              \
      typename Backend::template pointer_type<DTYPE const> & input, \
      typename Backend::template pointer_type<DTYPE> & workspace,   \
      typename Backend::template pointer_type<DTYPE> & output,      \
      SoftmaxParams const& params, Backend& backend)

#if defined(SNN_TEST_SYCLBLAS)
INSTANTIATE_LAUNCH(float, sycldnn::softmax::Softmax,
                   sycldnn::backend::SyclBLASBackend);
#else
INSTANTIATE_LAUNCH(float, sycldnn::softmax::Softmax,
                   sycldnn::backend::SNNBackend);
#endif

#ifdef SNN_USE_HALF
#if defined(SNN_TEST_SYCLBLAS)
INSTANTIATE_LAUNCH(cl::sycl::half, sycldnn::softmax::Softmax,
                   sycldnn::backend::SyclBLASBackend);
#else
INSTANTIATE_LAUNCH(cl::sycl::half, sycldnn::softmax::Softmax,
                   sycldnn::backend::SNNBackend);
#endif
#endif

#ifdef SNN_USE_DOUBLE
#if defined(SNN_TEST_SYCLBLAS)
INSTANTIATE_LAUNCH(double, sycldnn::softmax::Softmax,
                   sycldnn::backend::SyclBLASBackend);
#else
INSTANTIATE_LAUNCH(double, sycldnn::softmax::Softmax,
                   sycldnn::backend::SNNBackend);
#endif
#endif

}  // namespace internal
}  // namespace softmax
}  // namespace sycldnn
