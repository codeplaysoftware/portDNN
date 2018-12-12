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

#include <executors/executor_sycl.hpp>
#include <interface/blas3_interface.hpp>
#include <interface/blas_interface_sycl.hpp>

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
  using pointer_type = T*;

  /**
   * The internal pointer type for SyclBLASBackend.
   */
  template <typename T>
  using internal_pointer_type = T*;
};

/**
 * SyclBLAS backend for SYCL-DNN.
 *
 * Provides pointer handling and matrix multiplies using SyclBLAS.
 */
struct SyclBLASBackend final {
 private:
  blas::Executor<SYCL> executor_;

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
   * Gets a descriptive name for this backend.
   * \return a descriptive name for this backend.
   */
  char const* name() const { return "SyclBLASBackend"; }

  /**
   * Gets the SYCL queue that the backend is bound to.
   * \return Returns the SYCL queue that the backend is bound to.
   */
  cl::sycl::queue get_queue() { return executor_.get_queue(); }

  /**
   * Get a const reference to the SyclBLAS executor used in this backend.
   * \return A const reference to the SyclBLAS SyclDevice.
   */
  blas::Executor<SYCL> const& get_executor() const { return executor_; }

  /**
   * Get a reference to the SyclBLAS SyclDevice used in this backend.
   * \return A reference to the SyclBLAS SyclDevice.
   */
  blas::Executor<SYCL>& get_executor() { return executor_; }

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
  void release_internal_pointer(T* ptr) {
    SNN_UNUSED_VAR(ptr);
  }

  /**
   * Get a SYCL buffer from an external pointer.
   * \param ptr The pointer for which to retrieve the corresponding SYCL buffer.
   * \param n_elems The number of elements in the buffer. Unused in this case.
   * \return Returns a SYCL buffer corresponding to ptr.
   */
  template <typename T>
  auto get_buffer(pointer_type<T> ptr, size_t n_elems)
      -> decltype(this->executor_.get_buffer(ptr)) {
    SNN_UNUSED_VAR(n_elems);
    return executor_.get_buffer(ptr);
  }

  /**
   * Get a SYCL buffer from an internal pointer.
   * \param ptr The pointer for which to retrieve the corresponding SYCL buffer.
   * \param n_elems The number of elements in the buffer. Unused in this case.
   * \return Returns a SYCL buffer corresponding to ptr.
   */
  template <typename T>
  auto get_buffer_internal(internal_pointer_type<T> ptr, size_t n_elems)
      -> decltype(this->executor_.get_buffer(ptr)) {
    SNN_UNUSED_VAR(n_elems);
    return executor_.get_buffer(ptr);
  }

  /**
   * Return the offset from the start of the buffer.
   * \param ptr The pointer for which to retrieve the offset from the base of
   *            the corresponding SYCL buffer.
   * \return Returns the offset from the base of the SYCL buffer.
   */
  template <typename T>
  size_t get_offset(pointer_type<T> ptr) {
    return get_offset_internal(to_internal_pointer<T>(ptr));
  }

  /**
   * Return the offset from the start of the buffer.
   * \param ptr The pointer for which to retrieve the offset from the base of
   *            the corresponding SYCL buffer.
   * \return Returns the offset from the base of the SYCL buffer.
   */
  template <typename T>
  size_t get_offset_internal(internal_pointer_type<T> ptr) {
    return executor_.get_offset(ptr);
  }

  /**
   * Allocate a temporary buffer of the requested size.
   * \param n_bytes The size of the buffer in bytes.
   * \return Returns an internal pointer representing the new allocation.
   */
  template <typename T>
  internal_pointer_type<T> allocate(size_t n_bytes) {
    return executor_.allocate<T>(n_bytes / sizeof(T));
  }

  /**
   * Deallocate a temporary buffer.
   * \param ptr The pointer representing the buffer to deallocate.
   */
  template <typename T>
  void deallocate(internal_pointer_type<T> ptr) {
    executor_.deallocate(ptr);
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

    auto ldc = trans_m;
    auto lda = TransposeRHS ? k : trans_m;
    auto ldb = TransposeLHS ? trans_n : k;
    cl::sycl::event e = blas::_gemm(
        executor_, TransposeRHS ? 't' : 'n', TransposeLHS ? 't' : 'n', trans_m,
        trans_n, k, 1.0f, const_cast<T*>(rhs), lda, const_cast<T*>(lhs), ldb,
        beta, output, ldc);
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
    Index const lhs_size = m * k;
    Index const rhs_size = k * n;
    Index const out_size = m * n;

    cl::sycl::event event;
    for (int i = 0; i < n_batches; ++i) {
      Index const lhs_offset = lhs_size * i;
      Index const rhs_offset = rhs_size * i;
      Index const out_offset = out_size * i;
      event = matmul<TransposeLHS, TransposeRHS>(
          lhs + lhs_offset, rhs + rhs_offset, output + out_offset,
          static_cast<T>(0), m, k, n);
    }
    return event;
  }
};

}  // namespace backend
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_BACKEND_SYCL_BLAS_BACKEND_H_
