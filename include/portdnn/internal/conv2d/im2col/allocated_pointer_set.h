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
#ifndef PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_ALLOCATED_POINTER_SET_H_
#define PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_ALLOCATED_POINTER_SET_H_

#include "portdnn/conv2d/params.h"

#include "portdnn/internal/conv2d/alloc_info.h"

#include "portdnn/internal/conv2d/im2col/full_pointer_set.h"
#include "portdnn/internal/conv2d/im2col/transform_sizes.h"

#include "portdnn/internal/helpers/allocated_pointer.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace im2col {

/**
 * Set of all pointers required for im2col.
 *
 * Will allocate a temporary buffer for the input transform on construction,
 * which will be automatically deallocated on destruction.
 */
template <typename T, typename Backend, typename ConvType>
struct AllocatedPointerSet {
  using ConstPointer =
      typename Backend::template internal_pointer_type<T const>;
  using Pointer = typename Backend::template internal_pointer_type<T>;
  using AllocatedPointer =
      ::sycldnn::internal::helpers::AllocatedPointer<T, Backend>;

  AllocatedPointerSet(InternalPointerSet<T, Backend> const& set,
                      size_t size_per_image, Conv2DParams const& params,
                      Backend& backend)
      : allocated_transform_size{get_transform_size(size_per_image, params,
                                                    backend)},
        input{set.input.get()},
        filter{set.filter.get()},
        transform{sizeof(T) * allocated_transform_size, backend},
        output{set.output.get()} {}

  FullPointerSet<T, Backend, ConvType> to_full_pointer_set() {
    return {input, filter, transform.get(), output};
  }

  /** Add events to pointer on which to wait for before releasing memory */
  inline void pass_event_to_ptrs(const cl::sycl::event& event) {
    transform.set_event(event);
  }

  size_t allocated_transform_size;
  ConstPointer input;
  ConstPointer filter;
  AllocatedPointer transform;
  Pointer output;

 private:
  static size_t get_transform_size(size_t size_per_image,
                                   Conv2DParams const& params,
                                   Backend& backend) {
    auto queue = backend.get_queue();
    auto device = queue.get_device();

    size_t const alloc_limit =
        device.template get_info<cl::sycl::info::device::max_mem_alloc_size>() /
        sizeof(T);

    auto const transform_sizes = get_transform_sizes<ConvType>(params);
    auto const alloc_size_per_image =
        size_per_image + transform_sizes.output_transform_size;

    SNN_ASSERT(alloc_size_per_image + transform_sizes.filter_transform_size <
                   alloc_limit,
               "There is not enough available memory to safely allocate "
               "transformation memory for a single image");

    size_t const images_per_alloc =
        std::min((alloc_limit - transform_sizes.filter_transform_size) /
                     alloc_size_per_image,
                 static_cast<size_t>(params.batch));
    return images_per_alloc * alloc_size_per_image +
           transform_sizes.filter_transform_size;
  }
};

/**
 * Set of all pointers required for input backprop.
 *
 * Will allocate temporary buffers for the filter transform and the input
 * transform on construction, which will be automatically deallocated on
 * destruction.
 */
template <typename T, typename Backend>
struct AllocatedPointerSet<T, Backend, conv_type::InputBackprop> {
  using ConstPointer =
      typename Backend::template internal_pointer_type<T const>;
  using Pointer = typename Backend::template internal_pointer_type<T>;
  using AllocatedPointer =
      ::sycldnn::internal::helpers::AllocatedPointer<T, Backend>;

  AllocatedPointerSet(InternalPointerSet<T, Backend> const& set,
                      size_t size_per_image, Conv2DParams const& params,
                      Backend& backend)
      : allocated_transform_size{get_transform_size(size_per_image,
                                                    params.batch, backend)},
        input{set.input.get()},
        original_filter{set.filter.get()},
        filter{
            sizeof(T) * filter_transform_size<conv_type::InputBackprop>(params),
            backend},
        transform{sizeof(T) * allocated_transform_size, backend},
        output{set.output.get()} {}

  FullPointerSet<T, Backend, conv_type::InputBackprop> to_full_pointer_set() {
    return {input, original_filter, filter.get(), transform.get(), output};
  }

  /** Add events to pointer on which to wait for before releasing memory */
  inline void pass_event_to_ptrs(const cl::sycl::event& event) {
    filter.set_event(event);
    transform.set_event(event);
  }

  size_t allocated_transform_size;
  ConstPointer input;
  ConstPointer original_filter;
  AllocatedPointer filter;
  AllocatedPointer transform;
  Pointer output;

 private:
  /** Get the number of elements in the temporary transform tensor. */
  static size_t get_transform_size(size_t size_per_image, size_t n_images,
                                   Backend& backend) {
    cl::sycl::queue queue = backend.get_queue();
    cl::sycl::device device = queue.get_device();
    auto const alloc_info =
        get_alloc_info(device, n_images, size_per_image * sizeof(T));
    return size_per_image * alloc_info.images_per_alloc;
  }
};

}  // namespace im2col
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_ALLOCATED_POINTER_SET_H_
