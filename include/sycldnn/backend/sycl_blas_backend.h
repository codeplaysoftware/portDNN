/*
 * Copyright 2018 Codeplay Software Ltd
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
#ifndef SYCLDNN_INCLUDE_BACKEND_SYCL_BLAS_BACKEND_H_
#define SYCLDNN_INCLUDE_BACKEND_SYCL_BLAS_BACKEND_H_

/**
 * \file
 * Contains the implementation of \ref sycldnn::backend::SyclBLASBackend,
 * which provides pointer handling and matrix multiplies via SyclBLAS.
 */
#include "sycldnn/backend/backend_traits.h"
#include "sycldnn/helpers/macros.h"

#include "sycldnn/mem_object.h"

#include <sycl_blas.hpp>

namespace sycldnn {
namespace backend {

// Forward declaration to allow the BackendTraits specialisation.
struct SyclBLASBackend;

/**
 * The template specialisation of \ref
 * sycldnn::backend::BackendTraits<SyclBLASBackend>.
 *
 * Provides the pointer types for the SyclBLASBackend.
 */
template <>
struct BackendTraits<SyclBLASBackend> {
  /**
   * The external pointer type for SyclBLASBackend.
   */
  template <typename T>
  using pointer_type = blas::BufferIterator<T, blas::codeplay_policy>;

  /**
   * The internal pointer type for SyclBLASBackend.
   */
  template <typename T>
  using internal_pointer_type = blas::BufferIterator<T, blas::codeplay_policy>;
};

/**
 * SyclBLAS backend for SYCL-DNN.
 *
 * Provides pointer handling and matrix multiplies using SyclBLAS.
 */
struct SyclBLASBackend final {
 private:
  /** Alias for the SYCL-BLAS executor type. */
  using Executor = blas::Executor<blas::PolicyHandler<blas::codeplay_policy>>;
  Executor executor_;

 public:
  /** The pointer type used in interface of the SyclBLASBackend. */
  template <typename T>
  using pointer_type =
      typename BackendTraits<SyclBLASBackend>::template pointer_type<T>;
  /** The internal pointer type used internally by the SyclBLASBackend. */
  template <typename T>
  using internal_pointer_type = typename BackendTraits<
      SyclBLASBackend>::template internal_pointer_type<T>;

  /**
   * Constructs an instance of \ref sycldnn::backend::SyclBLASBackend from a
   * SYCL queue.
   * \param queue The SYCL queue to construct the backend from.
   */
  SyclBLASBackend(const cl::sycl::queue& queue) : executor_{queue} {}

  /**
   * Deleted copy constructor.
   */
  SNN_DISABLE_COPY(SyclBLASBackend);
  /**
   * Deleted move constructor.
   */
  SNN_DISABLE_MOVE(SyclBLASBackend);

  /**
   * Gets a descriptive name for this backend.
   * \return a descriptive name for this backend.
   */
  static char const* name() { return "SyclBLASBackend"; }

  /**
   * Gets the SYCL queue that the backend is bound to.
   * \return Returns the SYCL queue that the backend is bound to.
   */
  cl::sycl::queue get_queue() {
    return executor_.get_policy_handler().get_queue();
  }

  /**
   * Get a const reference to the SyclBLAS executor used in this backend.
   * \return A const reference to the SyclBLAS executor.
   */
  Executor const& get_executor() const { return executor_; }

  /**
   * Get a reference to the SyclBLAS executor used in this backend.
   * \return A reference to the SyclBLAS executor.
   */
  Executor& get_executor() { return executor_; }

  /**
   * Maps from external to internal pointer representations. This is a no-op
   * for the SyclBLAS backend.
   * \param ptr The external pointer to transform to the corresponding internal
   *            pointer representation.
   * \return Returns a pointer representation compatible with SyclBLAS.
   */
  template <typename T>
  internal_pointer_type<T> to_internal_pointer(pointer_type<T> ptr) {
    return ptr;
  }

  /**
   * Release the internal pointer, which has previously been returned from \ref
   * sycldnn::backend::SyclBLASBackend::to_internal_pointer.
   *
   * In this case it is a no-op.
   *
   * \param ptr The internal pointer to release.
   */
  template <typename T>
  void release_internal_pointer(internal_pointer_type<T> ptr) {
    SNN_UNUSED_VAR(ptr);
  }

  /**
   * Get a MemObject containing the buffer corresponding to a given pointer.
   * \param ptr     A pointer referring to a SYCL buffer with some offset.
   * \param n_elems The number of elements required within the MemObject.
   * \return Returns a MemObject corresponding to the pointer.
   */
  template <typename T>
  auto get_mem_object(pointer_type<T> ptr, size_t n_elems)
      -> decltype(make_mem_object(
          this->executor_.get_policy_handler().get_buffer(ptr).get_buffer(),
          n_elems, 0u)) {
    return make_mem_object(ptr.get_buffer(), n_elems, ptr.get_offset());
  }

  /** \copydoc get_mem_object */
  template <typename T>
  auto get_mem_object_internal(internal_pointer_type<T> ptr, size_t n_elems)
      -> decltype(make_mem_object(
          this->executor_.get_policy_handler().get_buffer(ptr).get_buffer(),
          n_elems, 0u)) {
    return make_mem_object(ptr.get_buffer(), n_elems, ptr.get_offset());
  }

  /**
   * Allocate a temporary buffer of the requested size.
   * \param n_bytes The size of the buffer in bytes.
   * \return Returns an internal pointer representing the new allocation.
   */
  template <typename T>
  internal_pointer_type<T> allocate(size_t n_bytes) {
    return blas::make_sycl_iterator_buffer<T, size_t>(n_bytes);
  }

  /**
   * Deallocate a temporary buffer. RAII means drop this.
   * \param ptr The pointer representing the buffer to deallocate.
   */
  template <typename T>
  void deallocate(internal_pointer_type<T> ptr) {
    SNN_UNUSED_VAR(ptr);
  }

  /**
   * A wrapper around a call to GEMM.
   *
   * Should perform the matrix multiply operation:
   *   output = lhs * rhs + beta * output
   * where lhs is a [m x k] matrix, rhs is a [k x n] matrix. The `bool`
   * template parameters determine whether or not to transpose the matrices.
   *
   * The matrices provided here are assumed to be in row-major ordering.Typical
   * BLAS implementations assume the matrices are column-major, so the
   * implementation of this method switches the order of `lhs` and
   * `rhs` to convert to row-major format.
   *
   * \param [in]     lhs       Pointer to a buffer containing the LHS matrix.
   * \param [in]     rhs       Pointer to a buffer containing the RHS matrix.
   * \param [in,out] output    Pointer to a buffer containing the output matrix.
   * \param [in]     beta      Scale multiplier for the output matrix.
   * \param [in]     m         Number of rows in the LHS matrix.
   * \param [in]     k         Number of columns in the LHS matrix and rows in
   *                           the RHS matrix.
   * \param [in]     n         Number of columns in the RHS matrix.
   *
   * \return A SYCL event corresponding to the matmul kernel launch.
   */
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
  cl::sycl::event matmul(internal_pointer_type<const T> const lhs,
                         internal_pointer_type<const T> const rhs,
                         internal_pointer_type<T> const output, T const beta,
                         Index const m, Index const k, Index const n) {
    // We are flipping the lhs/rhs, so we need to flip m/n
    auto trans_m = n;
    auto trans_n = m;

    cl::sycl::event e;
    if (m == 1) {
      // The LHS matrix is actually a vector
      auto gemv_m = TransposeRHS ? k : trans_m;
      auto gemv_n = TransposeRHS ? trans_m : k;
      auto gemv_lda = gemv_m;
      constexpr Index increment = 1;
      e = blas::_gemv(executor_, TransposeRHS ? 't' : 'n', gemv_m, gemv_n,
                      static_cast<T>(1), rhs, gemv_lda, lhs, increment, beta,
                      output, increment)
              .back();
    } else if (n == 1) {
      // The RHS matrix is actually a vector
      auto gemv_m = TransposeLHS ? trans_n : k;
      auto gemv_n = TransposeLHS ? k : trans_n;
      auto gemv_lda = gemv_m;
      constexpr Index increment = 1;
      e = blas::_gemv(executor_, TransposeLHS ? 'n' : 't', gemv_m, gemv_n,
                      static_cast<T>(1), lhs, gemv_lda, rhs, increment, beta,
                      output, increment)
              .back();
    } else {
      auto ldc = trans_m;
      auto lda = TransposeRHS ? k : trans_m;
      auto ldb = TransposeLHS ? trans_n : k;
      e = blas::_gemm(executor_, TransposeRHS ? 't' : 'n',
                      TransposeLHS ? 't' : 'n', trans_m, trans_n, k,
                      static_cast<T>(1), rhs, lda, lhs, ldb, beta, output, ldc)
              .back();
    }
    return e;
  }

  /**
   * Compute a batch of matrix multiplies.
   *
   * Assumes that lhs is a [batch x m x k] tensor and rhs is a [batch x k x n]
   * tensor.
   * Should perform the batched matrix multiply operation:
   *   output[i] = lhs[i] * rhs[i]
   * for 0 <= i < batch. Each matrix is assumed to be contiguous in memory and
   * in row-major format. The `bool` template parameters determine whether or
   * not to transpose the matrices.
   *
   * \param [in]     lhs       Pointer to a buffer containing the LHS matrix.
   * \param [in]     rhs       Pointer to a buffer containing the RHS matrix.
   * \param [in,out] output    Pointer to a buffer containing the output matrix.
   * \param [in]     n_batches The number of matrices in the batch.
   * \param [in]     m         Number of rows in the LHS matrix.
   * \param [in]     k         Number of columns in the LHS matrix and rows in
   *                           the RHS matrix.
   * \param [in]     n         Number of columns in the RHS matrix.
   *
   * \return A SYCL event corresponding to the matmul kernel launch.
   */
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
  cl::sycl::event batch_matmul(internal_pointer_type<const T> const lhs,
                               internal_pointer_type<const T> const rhs,
                               internal_pointer_type<T> const output,
                               Index const n_batches, Index const m,
                               Index const k, Index const n) {
    auto trans_m = n;
    auto trans_n = m;

    auto ldc = trans_m;
    auto lda = TransposeRHS ? k : trans_m;
    auto ldb = TransposeLHS ? trans_n : k;
    cl::sycl::event e =
        blas::_gemm_batched(executor_, TransposeRHS ? 't' : 'n',
                            TransposeLHS ? 't' : 'n', trans_m, trans_n, k,
                            static_cast<T>(1), rhs, lda, lhs, ldb,
                            static_cast<T>(0), output, ldc, n_batches)
            .back();
    return e;
  }
};

}  // namespace backend
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_BACKEND_SYCL_BLAS_BACKEND_H_
