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
#ifndef PORTDNN_INCLUDE_BACKEND_CLBLAST_BACKEND_H_
#define PORTDNN_INCLUDE_BACKEND_CLBLAST_BACKEND_H_

/**
 * \file
 * Contains the implementation of the \ref sycldnn::backend::CLBlastBackend,
 * which allocates memory via SYCL and does GEMM and GEMV operations with
 * the CLBlast library.
 */
#include "portdnn/backend/backend_traits.h"
#include "portdnn/backend/common_backend.h"
#include "portdnn/backend/device_mem_pointer.h"
#include "portdnn/backend/snn_reduce_provider.h"
#include "portdnn/helpers/macros.h"

#include "portdnn/mem_object.h"

#include <CL/cl.h>
#include <clblast.h>
#include <CL/sycl.hpp>
#include <SYCL/codeplay.hpp>

namespace sycldnn {
namespace backend {

// Forward declaration to allow the BackendTraits specialisation.
class CLBlastBackend;

/** Specialisation of \ref BackendTraits for the CLBlast backend. */
template <>
struct BackendTraits<CLBlastBackend> {
  /** External pointer type exposed by CLBlastBackend - same as internal. */
  template <typename T>
  using pointer_type = DeviceMemPointer<T>;

  /** Internal pointer type used in CLBlastBackend - same as external. */
  template <typename T>
  using internal_pointer_type = DeviceMemPointer<T>;
};

/**
 * \brief CLBlast backend for portDNN.
 *
 * Provides pointer handling and matrix multiplies using CLBlast.
 */
class CLBlastBackend final : public CommonBackend,
                             public SNNReduceProvider<CLBlastBackend> {
  /** Copy of SYCL queue that wraps the cl_command_queue used by CLBlast. */
  cl::sycl::queue queue_;
  /** Cached OpenCL command queue from SYCL queue, used in CLBlast API. */
  cl_command_queue cl_queue_;

 public:
  /** The pointer type used in interface of the CLBlastBackend. */
  template <typename T>
  using pointer_type =
      typename BackendTraits<CLBlastBackend>::template pointer_type<T>;
  /** The internal pointer type used internally by the CLBlastBackend. */
  template <typename T>
  using internal_pointer_type =
      typename BackendTraits<CLBlastBackend>::template internal_pointer_type<T>;

  /**
   * Constructs an instance of \ref sycldnn::backend::CLBlastBackend from a
   * SYCL queue. Retains the underlying `cl_command_queue` which is
   * released on destruction.
   * \param queue The SYCL queue to construct the backend from.
   */
  CLBlastBackend(cl::sycl::queue& queue)
      : CommonBackend(queue), queue_{queue}, cl_queue_{queue.get()} {}

  /** Explicit destructor releases cl_queue_ */
  ~CLBlastBackend() { clReleaseCommandQueue(cl_queue_); }

  /**
   * Deleted copy constructor.
   */
  SNN_DISABLE_COPY(CLBlastBackend);
  /**
   * Deleted move constructor.
   */
  SNN_DISABLE_MOVE(CLBlastBackend);

  /**
   * Gets a descriptive name for this backend.
   * \return a descriptive name for this backend.
   */
  char const* name() const { return "CLBlastBackend"; }

  /**
   * Gets the SYCL queue that the backend is bound to.
   * \return Returns the SYCL queue that the backend is bound to.
   */
  cl::sycl::queue get_queue() { return queue_; }

  /**
   * Conversion function from external to internal pointer representation.
   * Is a no-op for CLBlastBackend.
   * \param ptr Pointer to convert from
   * \return The passed-in parameter
   */
  template <typename T>
  internal_pointer_type<T> to_internal_pointer(pointer_type<T> ptr) {
    return ptr;
  }

  /**
   * Explicit release function for device memory. Is a no-op for this
   * backend.
   * \param ptr The pointer to deallocate
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
      -> decltype(make_mem_object(ptr.get_buffer(), 0ull, 0ull)) {
    return make_mem_object(ptr.get_buffer(), n_elems, ptr.get_offset());
  }

  /** \copydoc get_mem_object */
  template <typename T>
  auto get_mem_object_internal(internal_pointer_type<T> ptr, size_t n_elems)
      -> decltype(make_mem_object(ptr.get_buffer(), 0ull, 0ull)) {
    return make_mem_object(ptr.get_buffer(), n_elems, ptr.get_offset());
  }

  /**
   * Allocation function that creates an internal_pointer representing
   * memory on the device associated with queue_.
   * \param n_bytes The number of bytes to allocate on device
   * \return Pointer to the allocation
   */
  template <typename T>
  internal_pointer_type<T> allocate(size_t n_bytes) {
    return internal_pointer_type<T>{n_bytes / sizeof(T)};
  }

  /**
   * Deallocate a device pointer.
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
                         Index const m, Index const k, Index const n,
                         const std::vector<cl::sycl::event>& = {}) {
    using namespace clblast;
    auto a_buf = lhs.get_buffer();
    auto b_buf = rhs.get_buffer();
    auto o_buf = output.get_buffer();
    auto a_offset = lhs.get_offset();
    auto b_offset = rhs.get_offset();
    auto o_offset = output.get_offset();

    auto ev = queue_.submit([&](cl::sycl::codeplay::handler& cgh) {
      using namespace cl::sycl::access;
      auto a_acc = a_buf.template get_access<mode::read>(cgh);
      auto b_acc = b_buf.template get_access<mode::read>(cgh);
      auto o_acc = o_buf.template get_access<mode::read_write>(cgh);

      cgh.interop_task([=](cl::sycl::codeplay::interop_handle const& han) {
        auto a = han.get(a_acc);
        auto b = han.get(b_acc);
        auto o = han.get(o_acc);

        auto lda = TransposeLHS ? m : k;
        auto ldb = TransposeRHS ? k : n;
        auto ldc = n;

        auto transa = TransposeLHS ? Transpose::kYes : Transpose::kNo;
        auto transb = TransposeRHS ? Transpose::kYes : Transpose::kNo;

        constexpr T alpha = static_cast<T>(1);

        cl_event e;
        clblast::StatusCode code;
        if (n == 1) {
          auto gemv_m = TransposeLHS ? k : m;
          auto gemv_n = TransposeLHS ? m : k;
          auto gemv_lda = gemv_n;
          constexpr size_t increment = 1;
          code = clblast::Gemv(clblast::Layout::kRowMajor, transa, gemv_m,
                               gemv_n, alpha, a, a_offset, gemv_lda, b,
                               b_offset, increment, beta, o, o_offset,
                               increment, &cl_queue_, &e);
        } else {
          code = clblast::Gemm(clblast::Layout::kRowMajor, transa, transb, m, n,
                               k, alpha, a, a_offset, lda, b, b_offset, ldb,
                               beta, o, o_offset, ldc, &cl_queue_, &e);
        }
        if (code != clblast::StatusCode::kSuccess) {
          std::string excep("Bad return code from CLBlast GEMM: ");
          throw std::runtime_error(excep +
                                   std::to_string(static_cast<int>(code)));
        }
        clWaitForEvents(1, &e);
      });
    });
    return ev;
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
   * \param [in]     lhs        Pointer to a buffer containing the LHS matrix.
   * \param [in]     rhs        Pointer to a buffer containing the RHS matrix.
   * \param [in,out] output     Pointer to a buffer containing the output
   *                            matrix.
   * \param [in]     n_batches  The number of matrices in the batch.
   * \param [in]     m          Number of rows in the LHS matrix.
   * \param [in]     k          Number of columns in the LHS matrix and rows in
   *                            the RHS matrix.
   * \param [in]     n          Number of columns in the RHS matrix.
   * \param [in]     batch_type Format indicating how the batches are layed out.
   *
   * \return A SYCL event corresponding to the matmul kernel launch.
   */
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
  cl::sycl::event batch_matmul(
      internal_pointer_type<const T> const lhs,
      internal_pointer_type<const T> const rhs,
      internal_pointer_type<T> const output, Index const n_batches,
      Index const m, Index const k, Index const n,
      sycldnn::BatchFormat const batch_type = sycldnn::BatchFormat::STRIDED,
      const std::vector<cl::sycl::event>& = {}) {
    if (batch_type != sycldnn::BatchFormat::STRIDED) {
      throw std::runtime_error(
          "CLBlast batch matmul only supports strided batch format.");
    }

    using namespace clblast;
    auto lda = TransposeLHS ? m : k;
    auto ldb = TransposeRHS ? k : n;
    auto ldc = n;
    auto transa = TransposeLHS ? Transpose::kYes : Transpose::kNo;
    auto transb = TransposeRHS ? Transpose::kYes : Transpose::kNo;
    auto a_buf = lhs.get_buffer();
    auto b_buf = rhs.get_buffer();
    auto o_buf = output.get_buffer();
    auto a_offset = lhs.get_offset();
    auto b_offset = rhs.get_offset();
    auto o_offset = output.get_offset();

    auto ev = queue_.submit([&](cl::sycl::codeplay::handler& cgh) {
      using namespace cl::sycl::access;
      auto a_acc = a_buf.template get_access<mode::read>(cgh);
      auto b_acc = b_buf.template get_access<mode::read>(cgh);
      auto o_acc = o_buf.template get_access<mode::read_write>(cgh);

      cgh.interop_task([=](cl::sycl::codeplay::interop_handle const& han) {
        auto a = han.get(a_acc);
        auto b = han.get(b_acc);
        auto o = han.get(o_acc);

        constexpr T alpha = static_cast<T>(1);
        constexpr T beta = static_cast<T>(0);
        cl_event e;
        auto code = clblast::GemmStridedBatched(
            clblast::Layout::kRowMajor, transa, transb, m, n, k, alpha, a,
            a_offset, lda, m * k, b, b_offset, ldb, k * n, beta, o, o_offset,
            ldc, m * n, n_batches, &cl_queue_, &e);
        if (code != clblast::StatusCode::kSuccess) {
          std::string excep("Bad return code from CLBlast batch GEMM: ");
          throw std::runtime_error(excep +
                                   std::to_string(static_cast<int>(code)));
        }
        clWaitForEvents(1, &e);
      });
    });
    return ev;
  }
};  // namespace backend

}  // namespace backend
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_BACKEND_SYCL_BLAS_BACKEND_H_
