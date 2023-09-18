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
#ifndef PORTDNN_INCLUDE_INTERNAL_CONV2D_ALLOC_INFO_H_
#define PORTDNN_INCLUDE_INTERNAL_CONV2D_ALLOC_INFO_H_

#include <CL/sycl.hpp>

namespace sycldnn {
namespace conv2d {
namespace internal {

/** Struct containing information about allocation limits on a device. */
struct AllocInfo {
  /**
   * The maximum number of bytes that can safely be allocated on the queried
   * SYCL device.
   */
  size_t alloc_limit;
  /**
   * The number of images which could be allocated in a safely allocated
   * buffer.
   */
  size_t images_per_alloc;
  /**
   * True if a single image is larger than the allocation limit, so even trying
   * to allocate one image may cause an allocation failure.
   */
  bool alloc_warning;
};

/**
 * Query the SYCL device to get the largest amount of memory that can be
 * allocated and the maximum number of images of size `alloc_size_per_image`
 * that can be accomodated in a buffer of that size.
 *
 * \param device               SYCL device to query.
 * \param max_n_images         The maximum number of images required to be
 *                             allocated.
 * \param alloc_size_per_image Number of bytes required per image.
 */
inline AllocInfo get_alloc_info(cl::sycl::device const& device,
                                size_t max_n_images,
                                size_t alloc_size_per_image) {
  size_t alloc_limit =
      device.get_info<cl::sycl::info::device::max_mem_alloc_size>() / 4;
  bool alloc_warning = false;
  if (alloc_size_per_image > alloc_limit) {
    // Required allocation size is too large to be safely allocated on the
    // device. Provide an allocation warning to the caller.
    alloc_limit = alloc_size_per_image + 1;
    alloc_warning = true;
  }
  size_t const images_per_alloc =
      std::min(max_n_images, alloc_limit / alloc_size_per_image);

  return AllocInfo{alloc_limit, images_per_alloc, alloc_warning};
}

}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_INTERNAL_CONV2D_ALLOC_INFO_H_
