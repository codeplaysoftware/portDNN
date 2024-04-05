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
#ifndef SYCLDNN_INCLUDE_BACKEND_COMMON_BACKEND_H_
#define SYCLDNN_INCLUDE_BACKEND_COMMON_BACKEND_H_

/**
 * \file
 * Contains common methods used by all backends.
 */

#include <CL/sycl.hpp>
#include <string>
#include <unordered_map>

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
  bool supports_subgroup() { return max_num_sub_groups_ > 0; }

  /**
   * \brief Returns the maximum size of the workgroup.
   *
   * \return maximum size of the workgroup.
   */
  size_t get_max_wg_size() { return max_wg_size_; }

  /**
   * \brief Returns the maximum possible size of the subgroup.
   * Note: a smaller size may be chosen when running the actual kernel.
   *
   * \return maximum size of the subgroup.
   */

  size_t get_max_sg_size() { return max_sg_size_; }

 protected:
  /**
   * \brief Cache some information on construction.
   *
   * \param queue SYCL queue.
   */
  explicit CommonBackend(cl::sycl::queue& queue) {
    auto device = queue.get_device();
    max_num_sub_groups_ =
        device.get_info<cl::sycl::info::device::max_num_sub_groups>();
    const size_t max_work_group_size =
        device.get_info<cl::sycl::info::device::max_work_group_size>();
    const auto max_work_item_sizes =
        device.get_info<cl::sycl::info::device::max_work_item_sizes<3> >();
    max_wg_size_ = std::min(max_work_item_sizes[2], max_work_group_size);
#if defined(NVIDIA_GPU) || defined(AMD_GPU)
    max_sg_size_ = 32;
#else
    const auto subgroup_sizes =
        device.get_info<cl::sycl::info::device::sub_group_sizes>();
    max_sg_size_ =
        *std::max_element(subgroup_sizes.begin(), subgroup_sizes.end());
#endif
  }

 private:
  size_t max_num_sub_groups_;
  size_t max_wg_size_;
  size_t max_sg_size_;
};

}  // namespace backend
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_BACKEND_COMMON_BACKEND_H_
