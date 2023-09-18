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
#error "This file is provided as a guideline and should not be used."
namespace sycldnn {
/**
 * The backend has three main parts:
 *
 *   - An external pointer interface which is responsible for providing access
 *     to SYCL buffers from external pointers.
 *
 *   - An internal interface which provides matrix multiply methods and ways to
 *     create and access SYCL buffers used in the matrix multiplies.
 *
 *   - A way of converting the external pointers to internal pointers if they
 *     are different.
 */
struct BackendInterface : public ExternalHandler,
                          public InternalHandler,
                          public ExternalToInternalConverter {
  /** Return a descriptive name for this backend. */
  static char const* name();
};

struct ExternalHandler {
  /**
   * Pointer type used in the external interface, and passed by the user.
   *
   * This pointer type will have to match the pointer type used by whichever
   * external framework is using the portDNN library.
   */
  template <typename T>
  using pointer_type = /* implementation defined */;
  /**
   * Return a MemObject containing the buffer corresponding to a given pointer.
   * NOTE: that the actual type returned here does not need to exactly match
   * this type signature, as the allocator is included in the MemObject type.
   */
  template <typename T>
  MemObject<T, Alloc> get_mem_object(pointer_type<T> p, size_t n_elems);
  /** Return the SYCL queue used by this backend. */
  cl::sycl::queue get_queue();
};
struct InternalHandler {
  /**
   * Pointer type used internally to portDNN.
   *
   * This pointer type must match the type required by the matmul
   * implementation provided by the backend. This is the pointer type used for
   * temporary buffers returned from `allocate`.
   */
  template <typename T>
  using internal_pointer_type = /* implementation defined */;
  /** Allocate a temporary buffer of the requested size. */
  template <typename T>
  internal_pointer_type<T> allocate(size_t n_bytes);
  /** Deallocate a temporary buffer. */
  template <typename T>
  void deallocate(internal_pointer_type<T> p);
  /**
   * Return a MemObject containing the buffer corresponding to a given pointer.
   * NOTE: that the actual type returned here does not need to exactly match
   * this type signature, as the allocator is included in the MemObject type.
   */
  template <typename T>
  MemObject<T, Alloc> get_mem_object_internal(internal_pointer_type<T> p,
                                              size_t n_elems);
  /**
   * A wrapper around a call to GEMM.
   *
   * Should perform the matrix multiply operation:
   *   output = lhs * rhs + beta * output
   * where lhs is a [m x k] matrix, rhs is a [k x n] matrix. The `bool`
   * template parameters determine whether or not to transpose the matrices.
   *
   * The matrices provided here are assumed to be in row-major ordering.
   * Typical BLAS implementations assume the matrices are column-major, so the
   * implementation of this method may require switching the order of `lhs` and
   * `rhs` to convert to row-major format.
   */
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
  cl::sycl::event matmul(internal_pointer_type<const T> const lhs,
                         internal_pointer_type<const T> const rhs,
                         internal_pointer_type<T> const output, T const beta,
                         Index const m, Index const k, Index const n);
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
   * If a fast batched matrix multiply kernel is available it should be used
   * here, otherwise it can fall back to calling `matmul` a number of times.
   */
  template <bool TransposeLHS, bool TransposeRHS, typename T, typename Index>
  cl::sycl::event batch_matmul(
      internal_pointer_type<const T> const lhs,
      internal_pointer_type<const T> const rhs,
      internal_pointer_type<T> const output, Index const n_batches,
      Index const m, Index const k, Index const n,
      sycldnn::BatchFormat const batch_type = sycldnn::BatchFormat::STRIDED);
  /**
   * A wrapper around a call to reduce.
   *
   * Perform a reduction using Op on the outer axis from an input:
   * [batch, outer, inner].
   */
  template <typename Op, typename T, typename Index>
  cl::sycl::event reduce(internal_pointer_type<const T> const input,
                         internal_pointer_type<T> const output, Index batch,
                         Index outer, Index inner);

  /**
   * Returns whether the backend can use subgroup operations.
   */
  bool supports_subgroup();
};
struct ExternalToInternalConverter {
  template <typename T>
  using pointer_type = /* implementation defined */;
  template <typename T>
  using internal_pointer_type = /* implementation defined */;
  /**
   * Convert an external pointer type into an internal pointer type.
   * \param ptr An external pointer to convert to an internal pointer.
   * \return An internal pointer corresponding to external pointer ptr.
   */
  template <typename T>
  internal_pointer_type<T> to_internal_pointer(pointer_type<T> ptr);
  /**
   * Release an internal pointer which was constructed by to_internal_pointer.
   * \param ptr An internal pointer to release.
   */
  template <typename T>
  void release_internal_pointer(internal_pointer_type<T> ptr);
};
}  // namespace sycldnn
