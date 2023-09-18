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
#ifndef PORTDNN_INCLUDE_BACKEND_COMMON_BACKEND_H_
#define PORTDNN_INCLUDE_BACKEND_COMMON_BACKEND_H_

/**
 * \file
 * Contains common methods used by all backends.
 */

#include <CL/sycl.hpp>
#include <string>
#include <unordered_map>

#include "portdnn/internal/helpers/types.h"

namespace sycldnn {
namespace backend {

/**
 * Provide common backend methods.
 * Caches some device informations that are not already cached by common SYCL
 * implementation.
 */
struct CommonBackend {
  /**
   * \brief Returns whether the backend can use subgroup operations.
   *
   * \return Whether the backend can use subgroup operations.
   */
  bool supports_subgroup() { return max_num_sub_groups > 0; }

  /**
   * \brief Get the map caching kernel's subgroup sizes.
   *
   * \return Map caching kernel's subgroup sizes.
   */
  sycldnn::internal::types::KernelSubgroupSizesMap&
  get_max_kernel_sub_group_sizes() {
    return max_kernel_sub_group_sizes;
  }

#ifndef SNN_DISABLE_SYCL_PROGRAM
  /**
   * \brief Get the cached program.
   *
   * \return Cached program.
   */
  cl::sycl::program get_program() { return program; }
#endif

 protected:
#ifndef SNN_DISABLE_SYCL_PROGRAM
  /**
   * \brief Cache some information on construction.
   *
   * \param queue SYCL queue.
   */

  explicit CommonBackend(cl::sycl::queue& queue)
      : max_kernel_sub_group_sizes(),
        program(queue.get_context()),
        max_num_sub_groups() {
    auto device = queue.get_device();
    max_num_sub_groups =
        device.get_info<cl::sycl::info::device::max_num_sub_groups>();
  }
#else
  explicit CommonBackend(cl::sycl::queue& queue)
      : max_kernel_sub_group_sizes(), max_num_sub_groups() {
    auto device = queue.get_device();
    max_num_sub_groups =
        device.get_info<cl::sycl::info::device::max_num_sub_groups>();
  }
#endif

 private:
  sycldnn::internal::types::KernelSubgroupSizesMap max_kernel_sub_group_sizes;
#ifndef SNN_DISABLE_SYCL_PROGRAM
  cl::sycl::program program;
#endif
  size_t max_num_sub_groups;
};

}  // namespace backend
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_BACKEND_COMMON_BACKEND_H_
