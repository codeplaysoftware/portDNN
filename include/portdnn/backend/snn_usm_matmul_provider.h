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
#ifndef PORTDNN_INCLUDE_BACKEND_SNN_USM_MATMUL_PROVIDER_H_
#define PORTDNN_INCLUDE_BACKEND_SNN_USM_MATMUL_PROVIDER_H_

/**
 * \file
 * Contains the implementation of \ref sycldnn::backend::SNNUSMMatmulProvider,
 * which provides matmul and batch_matmul implementations using the internal
 * portDNN matmul kernels.
 */

#include "portdnn/backend/backend_helpers.h"
#include "portdnn/backend/backend_traits.h"
#include "portdnn/backend/internal_backend.h"
#include "portdnn/matmul/launch.h"
#include "portdnn/matmul/params.h"

namespace sycldnn {
namespace backend {

/**
 * CRTP module to provide matmul and batch_matmul implementations using the
 * internal portDNN kernels.
 */
template <typename Backend>
struct SNNUSMMatmulProvider {
 private:
  /** The pointer representation required by the internal handler. */
  template <typename T>
  using internal_pointer_type =
      typename BackendTraits<Backend>::template internal_pointer_type<T>;

 public:
  /**
   * A wrapper around a call to GEMM.
   *
   * Perform the matrix multiply operation:
   * \code
   *   output = lhs * rhs + beta * output
   * \endcode
   * where lhs is a [m x k] matrix, rhs is a [k x n] matrix. The `bool`
   * template parameters determine whether or not to transpose the matrices.
   * The matrices provided here are assumed to be in row-major ordering.
   *
   * \param [in]     lhs    Pointer to a buffer containing the LHS matrix.
   * \param [in]     rhs    Pointer to a buffer containing the RHS matrix.
   * \param [in,out] output Pointer to a buffer containing the output matrix.
   * \param [in]     beta   Scale multiplier for the output matrix.
   * \param [in]     m      Number of rows in the LHS matrix.
   * \param [in]     k      Number of columns in the LHS matrix and rows in the
   *                        RHS matrix.
   * \param [in]     n      Number of columns in the RHS matrix.
   * \param [in]     events Events which should be completed before the
   *                        operation
   *
   * \return A SYCL event corresponding to the matmul kernel launch.
   */
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index,
            typename U = Backend,
            typename = typename std::enable_if<
                sycldnn::backend::is_usm_backend_v<U>>::type>
  cl::sycl::event matmul(internal_pointer_type<const T> const lhs,
                         internal_pointer_type<const T> const rhs,
                         internal_pointer_type<T> const output, T const beta,
                         Index const m, Index const k, Index const n,
                         const std::vector<cl::sycl::event>& events = {}) {
    auto& underlying_backend = static_cast<Backend&>(*this);
    internal::InternalBackend<Backend> internal_backend{underlying_backend};
    auto status = matmul::launch<T, TransposeLHS, TransposeRHS>(
        lhs, rhs, output, sycldnn::matmul::MatmulParams{1, m, k, n, beta},
        internal_backend, events);
    SNN_ASSERT(status.status == StatusCode::OK,
               "Error launching matmul kernel.");
    return status.event;
  }
  /**
   * A wrapper around a call to GEMM.
   *
   * Perform the matrix multiply operation:
   * \code
   *   output = lhs * rhs + beta * output
   * \endcode
   * where lhs is a [m x k] matrix, rhs is a [k x n] matrix. The `bool`
   * template parameters determine whether or not to transpose the matrices.
   * The matrices provided here are assumed to be in row-major ordering.
   *
   * \param [in]     lhs    Pointer to a buffer containing the LHS matrix.
   * \param [in]     rhs    Pointer to a buffer containing the RHS matrix.
   * \param [in,out] output Pointer to a buffer containing the output matrix.
   * \param [in]     beta   Scale multiplier for the output matrix.
   * \param [in]     m      Number of rows in the LHS matrix.
   * \param [in]     k      Number of columns in the LHS matrix and rows in the
   *                        RHS matrix.
   * \param [in]     n      Number of columns in the RHS matrix.
   *
   * \return A SYCL event corresponding to the matmuul kernel launch.
   */
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index,
            typename U = Backend,
            typename = typename std::enable_if<
                sycldnn::backend::is_buffer_backend_v<U>>::type>
  cl::sycl::event matmul(internal_pointer_type<const T> const lhs,
                         internal_pointer_type<const T> const rhs,
                         internal_pointer_type<T> const output, T const beta,
                         Index const m, Index const k, Index const n) {
    auto& underlying_backend = static_cast<Backend&>(*this);
    internal::InternalBackend<Backend> internal_backend{underlying_backend};
    auto status = matmul::launch<T, TransposeLHS, TransposeRHS>(
        lhs, rhs, output, sycldnn::matmul::MatmulParams{1, m, k, n, beta},
        internal_backend);
    SNN_ASSERT(status.status == StatusCode::OK,
               "Error launching matmul kernel.");
    return status.event;
  }

  /**
   * Compute a batch of matrix multiplies.
   *
   * Perform the batched matrix multiply operation:
   * \code
   *   output[i] = lhs[i] * rhs[i]
   * \endcode
   * for 0 <= i < batch, where lhs is a [batch x m x k] tensor and rhs is a
   * [batch x k x n] tensor. Each matrix is assumed to be contiguous in memory
   * and in row-major format. The `bool` template parameters determine whether
   * or not to transpose the matrices.
   *
   * \param [in]     lhs        Pointer to a buffer containing the LHS matrix.
   * \param [in]     rhs        Pointer to a buffer containing the RHS matrix.
   * \param [in,out] output     Pointer to a buffer containing the output
   * matrix. \param [in]     n_batches  Scale multiplier for the output matrix.
   * \param [in]     m          Number of rows in the LHS matrix.
   * \param [in]     k          Number of columns in the LHS matrix and rows in
   *                            the RHS matrix.
   * \param [in]     n          Number of columns in the RHS matrix.
   * \param [in]     batch_type Format indicating how the batches are layed out.
   *
   * \return A SYCL event corresponding to the matmul kernel launch.
   */
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index,
            typename U = Backend,
            typename = typename std::enable_if<
                sycldnn::backend::is_buffer_backend_v<U>>::type>
  cl::sycl::event batch_matmul(
      internal_pointer_type<const T> const lhs,
      internal_pointer_type<const T> const rhs,
      internal_pointer_type<T> const output, Index const n_batches,
      Index const m, Index const k, Index const n,
      sycldnn::BatchFormat const batch_type = sycldnn::BatchFormat::STRIDED) {
    if (batch_type != sycldnn::BatchFormat::STRIDED) {
      throw std::runtime_error(
          "SNN batch matmul only supports strided batch format.");
    }

    auto& underlying_backend = static_cast<Backend&>(*this);
    internal::InternalBackend<Backend> internal_backend{underlying_backend};
    auto status = matmul::launch<T, TransposeLHS, TransposeRHS>(
        lhs, rhs, output,
        sycldnn::matmul::MatmulParams{n_batches, m, k, n, T{0}},
        internal_backend);
    SNN_ASSERT(status.status == StatusCode::OK,
               "Error launching matmul kernel.");
    return status.event;
  }

  /**
   * Compute a batch of matrix multiplies.
   *
   * Perform the batched matrix multiply operation:
   * \code
   *   output[i] = lhs[i] * rhs[i]
   * \endcode
   * for 0 <= i < batch, where lhs is a [batch x m x k] tensor and rhs is a
   * [batch x k x n] tensor. Each matrix is assumed to be contiguous in memory
   * and in row-major format. The `bool` template parameters determine whether
   * or not to transpose the matrices.
   *
   * \param [in]     lhs        Pointer to a buffer containing the LHS matrix.
   * \param [in]     rhs        Pointer to a buffer containing the RHS matrix.
   * \param [in,out] output     Pointer to a buffer containing the output
   * matrix. \param [in]     n_batches  Scale multiplier for the output matrix.
   * \param [in]     m          Number of rows in the LHS matrix.
   * \param [in]     k          Number of columns in the LHS matrix and rows in
   *                            the RHS matrix.
   * \param [in]     n          Number of columns in the RHS matrix.
   * \param [in]     batch_type Format indicating how the batches are layed out.
   * \param [in]     events     Events which should be completed before the
   *                            operation
   *
   * \return A SYCL event corresponding to the matmul kernel launch.
   */
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index,
            typename U = Backend,
            typename = typename std::enable_if<
                sycldnn::backend::is_usm_backend_v<U>>::type>
  cl::sycl::event batch_matmul(
      internal_pointer_type<const T> const lhs,
      internal_pointer_type<const T> const rhs,
      internal_pointer_type<T> const output, Index const n_batches,
      Index const m, Index const k, Index const n,
      sycldnn::BatchFormat const batch_type = sycldnn::BatchFormat::STRIDED,
      const std::vector<cl::sycl::event>& events = {}) {
    if (batch_type != sycldnn::BatchFormat::STRIDED) {
      throw std::runtime_error(
          "SNN batch matmul only supports strided batch format.");
    }

    auto& underlying_backend = static_cast<Backend&>(*this);
    internal::InternalBackend<Backend> internal_backend{underlying_backend};
    auto status = matmul::launch<T, TransposeLHS, TransposeRHS>(
        lhs, rhs, output,
        sycldnn::matmul::MatmulParams{n_batches, m, k, n, T{0}},
        internal_backend, events);
    SNN_ASSERT(status.status == StatusCode::OK,
               "Error launching matmul kernel.");
    return status.event;
  }
};

}  // namespace backend
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_BACKEND_SNN_USM_MATMUL_PROVIDER_H_
