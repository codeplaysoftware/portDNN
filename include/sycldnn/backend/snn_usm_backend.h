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
#ifndef SYCLDNN_INCLUDE_BACKEND_SNN_USM_BACKEND_H_
#define SYCLDNN_INCLUDE_BACKEND_SNN_USM_BACKEND_H_

#include "sycldnn/backend/common_backend.h"
#include "sycldnn/backend/device_mem_pointer.h"
#include "sycldnn/backend/snn_usm_matmul_provider.h"
#include "sycldnn/backend/snn_usm_reduce_provider.h"

#include <CL/sycl.hpp>
#include <numeric>

namespace sycldnn {
namespace backend {

// Forward declaration to allow the BackendTraits specialisation.
struct SNNUSMBackend;

/**
 * The template specialisation of \ref
 * sycldnn::backend::BackendTraits<SNNUSMBackend>.
 *
 * Provides the pointer types for the SNNUSMBackend.
 */
template <>
struct BackendTraits<SNNUSMBackend> {
  /**
   * The external pointer type for SNNUSMBackend.
   */
  template <typename T>
  using pointer_type = T*;

  /**
   * The internal pointer type for SNNUSMBackend.
   */
  template <typename T>
  using internal_pointer_type = T*;
};

/**
 * Standard test backend for SYCL-DNN.
 *
 * Provides pointer handling, matrix multiplies and reduce using our internal
 * kernels.
 */
struct SNNUSMBackend final : public CommonBackend,
                             public SNNUSMMatmulProvider<SNNUSMBackend>,
                             public SNNUSMReduceProvider<SNNUSMBackend> {
  /** The pointer type used in interface of the SNNUSMBackend. */
  template <typename T>
  using pointer_type =
      typename BackendTraits<SNNUSMBackend>::template pointer_type<T>;

  /** The internal pointer type used internally by the SNNUSMBackend. */
  template <typename T>
  using internal_pointer_type =
      typename BackendTraits<SNNUSMBackend>::template internal_pointer_type<T>;

  /**
   * Construct an SNNUSMBackend with the given queue. All SYCL-DNN operations
   * launched with this backend will be submitted to this queue.
   *
   * \param queue The SYCL queue to use with this backend.
   */
  SNNUSMBackend(cl::sycl::queue queue)
      : CommonBackend{queue}, queue_{std::move(queue)} {}

  /**
   * Allocate a tensor to be used internally.
   * \param n_elems The size of the allocation in number of elements.
   * \return Returns a pointer to allocation, using the internal pointer
   *         representation.
   * */
  template <typename T>
  internal_pointer_type<T> allocate(size_t n_elems) {
    return cl::sycl::malloc_device<T>(n_elems, queue_);
  }

  /**
   * Deallocate an internal tensor.
   * \param ptr A pointer to the allocation to deallocate.
   */
  template <typename T>
  void deallocate(internal_pointer_type<T> ptr) {
    cl::sycl::free(ptr, queue_);
  }

  /**
   * Get a USMMemObject containing the pointer.
   * \param ptr     Memory pointer.
   * \param n_elems The number of elements required within the MemObject.
   * \param offset The number of elements to offset ptr by.
   * \return Returns a USMMemObject corresponding to the pointer.
   */
  template <typename T>
  auto get_mem_object(pointer_type<T> ptr, size_t n_elems, size_t offset = 0)
      -> decltype(make_usm_mem_object<T>(ptr, n_elems, offset)) {
    return make_usm_mem_object<T>(ptr, n_elems, offset);
  }

  /** \copydoc get_mem_object */
  template <typename T>
  auto get_mem_object_internal(internal_pointer_type<T> ptr, size_t n_elems,
                               size_t offset = 0)
      -> decltype(make_usm_mem_object(ptr, n_elems, offset)) {
    return make_usm_mem_object(ptr, n_elems, offset);
  }

  /**
   * Maps from external to internal pointer representations. This is a no-op for
   * the SNN backend.
   * \param ptr The external pointer to transform to the corresponding internal
   *            pointer representation.
   * \return Returns an internal pointer representation compatible with \ref
   *         sycldnn::backend::SNNUSMBackend.
   */
  template <typename T>
  internal_pointer_type<T> to_internal_pointer(pointer_type<T> ptr) {
    return ptr;
  }

  /**
   * Release the internal pointer, which has previously been returned from \ref
   * sycldnn::backend::SNNUSMBackend::to_internal_pointer.
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
  static char const* name() { return "SNNUSMBackend"; }

  /** Allows the backend to fill to its specific memory
   * \param mem memory object containing buffer to fill
   * \param val value to fill to
   * \param size number of elements to set
   * \param dependecies vector of events to wait before running fill
   * \return an event of the submit of the fill
   */
  template <typename T>
  sycl::event fill(USMMemObject<T>& mem, size_t size, T val,
                   const std::vector<sycl::event>& dependecies = {}) {
    if (val == 0.f) {
      return queue_.submit([&](cl::sycl::handler& cgh) {
        cgh.depends_on(dependecies);
        auto out = mem.write_mem(cgh).get_pointer().get();
        cgh.memset(out, val, sizeof(T) * size);
      });
    } else {
      return queue_.submit([&](cl::sycl::handler& cgh) {
        cgh.depends_on(dependecies);
        auto out = mem.write_mem(cgh).get_pointer().get();
        cgh.fill<T>(out, val, size);
      });
    }
  }

  /** Allows the backend to set the contents its specific memory to value
   * \param mem USM pointer to be fill
   * \param val value to set memory to
   * \param size number of elements to set
   * \param dependecies vector of events to wait before running fill
   * \return an event of the submit of the fill
   */
  template <typename T>
  sycl::event fill(internal_pointer_type<T>& mem, size_t size, T val,
                   const std::vector<sycl::event>& dependecies = {}) {
    if (val == 0.f) {
      return queue_.submit([&](cl::sycl::handler& cgh) {
        cgh.depends_on(dependecies);
        cgh.memset(mem, val, sizeof(T) * size);
      });
    } else {
      return queue_.submit([&](cl::sycl::handler& cgh) {
        cgh.depends_on(dependecies);
        cgh.fill<T>(mem, val, size);
      });
    }
  }

  /** Allows the backend to copy to its specific memory
   * \param out_mem memory object containing buffer to copy
   * \param in_mem value to copy from
   * \param size number of elements to copy
   * \param dependecies vector of events to wait before running copy
   * \return an event of the submit of the copy
   */
  template <typename T>
  sycl::event copy(USMMemObject<const T>& in_mem, USMMemObject<T>& out_mem,
                   size_t size,
                   const std::vector<sycl::event>& dependecies = {}) {
    return queue_.submit([&](cl::sycl::handler& cgh) {
      cgh.depends_on(dependecies);
      auto in = in_mem.read_mem(cgh).get_pointer().get();
      auto out = out_mem.write_mem(cgh).get_pointer().get();
      cgh.copy(in, out, size);
    });
  }

 private:
  cl::sycl::queue queue_;
};

}  // namespace backend
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_BACKEND_SNN_USM_BACKEND_H_
