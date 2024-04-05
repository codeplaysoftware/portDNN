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

#include <CL/sycl.hpp>
#include <utility>

#include "src/batchnorm/kernels.h"
#include "src/batchnorm/queue_batchnorm_kernel.h"
#include "sycldnn/backend/snn_backend.h"
#include "sycldnn/backend/snn_usm_backend.h"
#include "sycldnn/export.h"
#include "sycldnn/internal/batchnorm/launch_internal.h"

namespace sycldnn {
namespace batchnorm {
namespace internal {

template <typename T, typename Backend, template <typename> class MemObj>
SNNStatus launch_inference_internal(
    MemObj<T const>& input, MemObj<T const>& beta, MemObj<T const>& gamma,
    MemObj<T const>& input_mean, MemObj<T const>& input_variance,
    MemObj<T>& output, BatchNormParams const& params, Backend& backend,
    const std::vector<cl::sycl::event>& events) {
  using Index = int;
  bool const is_nchw = params.input_format == sycldnn::DataFormat::NCHW;
  if ((params.rows * params.cols) % 4 == 0 && is_nchw) {
    return queue_batchnorm_inference<T, Index, 4, MemObj>(
        input, beta, gamma, input_mean, input_variance, output, params,
        backend.get_queue(), events);
  } else if ((params.rows * params.cols) % 2 == 0 && is_nchw) {
    return queue_batchnorm_inference<T, Index, 2, MemObj>(
        input, beta, gamma, input_mean, input_variance, output, params,
        backend.get_queue(), events);
  } else {
    return queue_batchnorm_inference<T, Index, 1, MemObj>(
        input, beta, gamma, input_mean, input_variance, output, params,
        backend.get_queue(), events);
  }
}

template <typename T, typename Backend, template <typename> class MemObj>
SNNStatus launch_training_internal(
    MemObj<T const>& input, MemObj<T const>& beta, MemObj<T const>& gamma,
    MemObj<T>& running_mean, MemObj<T>& running_variance, MemObj<T>& cache_mean,
    MemObj<T>& cache_invvariance, MemObj<T>& workspace_var, MemObj<T>& output,
    BatchNormParams const& params, Backend& backend,
    const std::vector<cl::sycl::event>& events) {
  return queue_batchnorm_training<T, int>(
      input, beta, gamma, running_mean, running_variance, cache_mean,
      cache_invvariance, workspace_var, output, params, backend.get_queue(),
      events);
}

template <typename T, typename Backend, template <typename> class MemObj>
SNNStatus launch_gradient_internal(
    MemObj<T const>& input, MemObj<T const>& out_grad, MemObj<T>& in_grad,
    MemObj<T const>& gamma, MemObj<T const>& cache_mean,
    MemObj<T const>& cache_invvariance, MemObj<T>& workspace,
    BatchNormParams const& params, Backend& backend,
    const std::vector<cl::sycl::event>& events) {
  using Index = int;
  bool const is_nchw = params.input_format == sycldnn::DataFormat::NCHW;
  if ((params.rows * params.cols) % 4 == 0 && is_nchw) {
    return queue_batchnorm_gradient<T, Index, 4>(
        input, out_grad, in_grad, gamma, cache_mean, cache_invvariance,
        workspace, params, backend.get_queue(), events);
  } else if ((params.rows * params.cols) % 2 == 0 && is_nchw) {
    return queue_batchnorm_gradient<T, Index, 2>(
        input, out_grad, in_grad, gamma, cache_mean, cache_invvariance,
        workspace, params, backend.get_queue(), events);
  } else {
    return queue_batchnorm_gradient<T, Index, 1>(
        input, out_grad, in_grad, gamma, cache_mean, cache_invvariance,
        workspace, params, backend.get_queue(), events);
  }
}

#define INSTANTIATE_LAUNCH(MEM_OBJ, BACKEND, DTYPE)                           \
  template SNN_EXPORT SNNStatus                                               \
  launch_inference_internal<DTYPE, backend::BACKEND, MEM_OBJ>(                \
      MEM_OBJ<DTYPE const> & input, MEM_OBJ<DTYPE const> & beta,              \
      MEM_OBJ<DTYPE const> & gamma, MEM_OBJ<DTYPE const> & input_mean,        \
      MEM_OBJ<DTYPE const> & input_variance, MEM_OBJ<DTYPE> & output,         \
      BatchNormParams const&, backend::BACKEND& backend,                      \
      const std::vector<cl::sycl::event>& events);                            \
  template SNN_EXPORT SNNStatus                                               \
  launch_training_internal<DTYPE, backend::BACKEND, MEM_OBJ>(                 \
      MEM_OBJ<DTYPE const> & input, MEM_OBJ<DTYPE const> & beta,              \
      MEM_OBJ<DTYPE const> & gamma, MEM_OBJ<DTYPE> & running_mean,            \
      MEM_OBJ<DTYPE> & running_variance, MEM_OBJ<DTYPE> & cache_mean,         \
      MEM_OBJ<DTYPE> & cache_invvariance, MEM_OBJ<DTYPE> & workspace_var,     \
      MEM_OBJ<DTYPE> & output, BatchNormParams const& params,                 \
      backend::BACKEND& backend, const std::vector<cl::sycl::event>& events); \
  template SNN_EXPORT SNNStatus                                               \
  launch_gradient_internal<DTYPE, backend::BACKEND, MEM_OBJ>(                 \
      MEM_OBJ<DTYPE const> & input, MEM_OBJ<DTYPE const> & out_grad,          \
      MEM_OBJ<DTYPE> & in_grad, MEM_OBJ<DTYPE const> & gamma,                 \
      MEM_OBJ<DTYPE const> & cache_mean,                                      \
      MEM_OBJ<DTYPE const> & cache_invvariance, MEM_OBJ<DTYPE> & workspace,   \
      BatchNormParams const& params, backend::BACKEND& backend,               \
      const std::vector<cl::sycl::event>& events);

INSTANTIATE_LAUNCH(BufferMemObject, SNNBackend, float)

#ifdef SNN_ENABLE_USM
INSTANTIATE_LAUNCH(USMMemObject, SNNUSMBackend, float)
#endif
#undef INSTANTIATE_LAUNCH

}  // namespace internal
}  // namespace batchnorm
}  // namespace sycldnn
