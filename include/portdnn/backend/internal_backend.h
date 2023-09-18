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
#ifndef PORTDNN_INCLUDE_BACKEND_INTERNAL_BACKEND_H_
#define PORTDNN_INCLUDE_BACKEND_INTERNAL_BACKEND_H_

#include "portdnn/backend/backend_traits.h"
#include "portdnn/internal/helpers/types.h"

namespace sycldnn {
namespace backend {
namespace internal {

/**
 * The portDNN matmul and reduce launchers use the backend's external pointer
 * type and buffer accessors, however the calls to Backend::matmul,
 * Backend::batch_matmul and Backend::reduce use the backend's internal pointer
 * type and buffer accessors. This means we need to create a new backend just to
 * provide access to the correct types and methods when calling launch.
 */
template <typename Backend>
struct InternalBackend {
  /** Forward internal_pointer_type to pointer_type */
  template <typename T>
  using pointer_type =
      typename BackendTraits<Backend>::template internal_pointer_type<T>;

  /**
   * Construct a InternalBackend which forwards buffer access calls to the
   * provided backend.
   *
   * \param backend Underlying backend which provides access to buffers.
   */
  explicit InternalBackend(Backend backend)
      : underlying_backend{std::move(backend)} {}

  /**
   * Get the mem object corresponding to the provided pointer type of the
   * specified size.
   *
   * \param [in] ptr Pointer into a SYCL buffer or pointer.
   * \param [in] n_elems Number of elements expected to be in the buffer.
   * \return Buffer corresponding to the provided pointer.
   */
  template <typename T>
  auto get_mem_object(pointer_type<T> ptr, size_t n_elems)
      -> decltype(std::declval<Backend>().get_mem_object(ptr, n_elems)) {
    return underlying_backend.get_mem_object(ptr, n_elems);
  }

  /**
   * \brief Get the underlying queue
   *
   * \return cl::sycl::queue Underlying queue
   */
  cl::sycl::queue get_queue() { return underlying_backend.get_queue(); }

  /**
   * \brief Returns whether the backend can use subgroup operations.
   *
   * \return Whether the backend can use subgroup operations.
   */
  bool supports_subgroup() { return underlying_backend.supports_subgroup(); }

  /**
   * \brief Get the map caching kernel's subgroup sizes.
   *
   * \return Map caching kernel's subgroup sizes.
   */
  sycldnn::internal::types::KernelSubgroupSizesMap&
  get_max_kernel_sub_group_sizes() {
    return underlying_backend.get_max_kernel_sub_group_sizes();
  }

#ifndef SNN_DISABLE_SYCL_PROGRAM
  /**
   * \brief Get the cached program.
   *
   * \return Cached program.
   */
  cl::sycl::program get_program() { return underlying_backend.get_program(); }
#endif

 private:
  Backend underlying_backend;
};

}  // namespace internal
}  // namespace backend
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_BACKEND_INTERNAL_BACKEND_H_
