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
#ifndef PORTDNN_INCLUDE_BACKEND_SNN_BACKEND_H_
#define PORTDNN_INCLUDE_BACKEND_SNN_BACKEND_H_

#include "portdnn/backend/common_backend.h"
#include "portdnn/backend/device_mem_pointer.h"
#include "portdnn/backend/snn_matmul_provider.h"
#include "portdnn/backend/snn_reduce_provider.h"

#include <CL/sycl.hpp>
#include <numeric>

namespace sycldnn {
namespace backend {

// Forward declaration to allow the BackendTraits specialisation.
struct SNNBackend;

/**
 * The template specialisation of \ref
 * sycldnn::backend::BackendTraits<SNNBackend>.
 *
 * Provides the pointer types for the SNNBackend.
 */
template <>
struct BackendTraits<SNNBackend> {
  /**
   * The external pointer type for SNNBackend.
   */
  template <typename T>
  using pointer_type = DeviceMemPointer<T>;

  /**
   * The internal pointer type for SNNBackend.
   */
  template <typename T>
  using internal_pointer_type = DeviceMemPointer<T>;
};

/**
 * Standard test backend for portDNN.
 *
 * Provides pointer handling, matrix multiplies and reduce using our internal
 * kernels.
 */
struct SNNBackend final : public CommonBackend,
                          public SNNMatmulProvider<SNNBackend>,
                          public SNNReduceProvider<SNNBackend> {
  /** The pointer type used in interface of the SNNBackend. */
  template <typename T>
  using pointer_type =
      typename BackendTraits<SNNBackend>::template pointer_type<T>;

  /** The internal pointer type used internally by the SNNBackend. */
  template <typename T>
  using internal_pointer_type =
      typename BackendTraits<SNNBackend>::template internal_pointer_type<T>;

  /**
   * Construct an SNNBackend with the given queue. All portDNN operations
   * launched with this backend will be submitted to this queue.
   *
   * \param queue The SYCL queue to use with this backend.
   */
  SNNBackend(cl::sycl::queue queue)
      : CommonBackend{queue}, queue_{std::move(queue)} {}

  /**
   * Allocate a tensor to be used internally.
   * \param n_elems The size of the allocation in number of elements.
   * \return Returns a pointer to allocation, using the internal pointer
   *         representation.
   * */
  template <typename T>
  internal_pointer_type<T> allocate(size_t n_elems) {
    return internal_pointer_type<T>{n_elems};
  }

  /**
   * Deallocate an internal tensor.
   * \param ptr A pointer to the allocation to deallocate.
   */
  template <typename T>
  void deallocate(internal_pointer_type<T> ptr) {
    SNN_UNUSED_VAR(ptr)
  }

  /**
   * Get a MemObject containing the buffer corresponding to a given pointer.
   * \param ptr     A pointer referring to a SYCL buffer with some offset.
   * \param n_elems The number of elements required within the MemObject.
   * \return Returns a MemObject corresponding to the pointer.
   */
  template <typename T>
  auto get_mem_object(pointer_type<T> ptr, size_t n_elems)
      -> decltype(make_buffer_mem_object(ptr.get_buffer(), n_elems,
                                         ptr.get_offset())) {
    return make_buffer_mem_object(ptr.get_buffer(), n_elems, ptr.get_offset());
  }

  /** \copydoc get_mem_object */
  template <typename T>
  auto get_mem_object_internal(internal_pointer_type<T> ptr, size_t n_elems)
      -> decltype(make_buffer_mem_object(ptr.get_buffer(), n_elems,
                                         ptr.get_offset())) {
    return make_buffer_mem_object(ptr.get_buffer(), n_elems, ptr.get_offset());
  }

  /**
   * Maps from external to internal pointer representations. This is a no-op for
   * the SNN backend.
   * \param ptr The external pointer to transform to the corresponding internal
   *            pointer representation.
   * \return Returns an internal pointer representation compatible with \ref
   *         sycldnn::backend::SNNBackend.
   */
  template <typename T>
  internal_pointer_type<T> to_internal_pointer(pointer_type<T> ptr) {
    return ptr;
  }

  /**
   * Release the internal pointer, which has previously been returned from \ref
   * sycldnn::backend::SNNBackend::to_internal_pointer.
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
   * Gets the SYCL queue that the backend is bound to.
   * \return Returns the SYCL queue that the backend is bound to.
   */
  cl::sycl::queue& get_queue() { return queue_; }

  /**
   * Gets a descriptive name for this backend.
   * \return a descriptive name for this backend.
   */
  static char const* name() { return "SNNBackend"; }

 private:
  cl::sycl::queue queue_;
};

}  // namespace backend
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_BACKEND_SNN_BACKEND_H_
