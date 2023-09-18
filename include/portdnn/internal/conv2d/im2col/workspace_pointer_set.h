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
#ifndef PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_WORKSPACE_POINTER_SET_H_
#define PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_WORKSPACE_POINTER_SET_H_

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
 * Will use the given workspace to provide any required temporary buffers.
 */
template <typename T, typename Backend, typename ConvType>
struct WorkspacePointerSet {
  using ConstPointer =
      typename Backend::template internal_pointer_type<T const>;
  using Pointer = typename Backend::template internal_pointer_type<T>;
  using InternalPointer =
      ::sycldnn::internal::helpers::InternalPointer<T, Backend>;

  WorkspacePointerSet(InternalPointerSet<T, Backend> const& set,
                      typename Backend::template pointer_type<T> workspace,
                      size_t size_per_image, Conv2DParams const& params,
                      size_t workspace_size, Backend& backend)
      : minibatch_size{get_minibatch_size(workspace_size, size_per_image,
                                          params)},
        input{set.input.get()},
        filter{set.filter.get()},
        transform{workspace, backend},
        output{set.output.get()} {}

  FullPointerSet<T, Backend, ConvType> to_full_pointer_set() {
    return {input, filter, transform.get(), output};
  }

  size_t minibatch_size;
  ConstPointer input;
  ConstPointer filter;
  InternalPointer transform;
  Pointer output;

 private:
  /** Get the size of minibatch to use for the given workspace size. */
  static size_t get_minibatch_size(size_t workspace_size, size_t size_per_image,
                                   Conv2DParams const& params) {
    auto const transform_sizes = get_transform_sizes<ConvType>(params);
    return (workspace_size - transform_sizes.filter_transform_size) /
           (size_per_image + transform_sizes.output_transform_size);
  }
};

/**
 * /copydoc WorkspacePointerSet
 *
 * Unlike the Forward and FilterBackprop passes, when computing the
 * InputBackprop the filter values must be rotated, and so an additional
 * temporary buffer is required to hold the transformed filter data. The
 * WorkspacePointerSet must set aside part of the workspace for this buffer, as
 * well as keeping the pointer to the original filter data for the filter
 * transform kernel.
 */
template <typename T, typename Backend>
struct WorkspacePointerSet<T, Backend, conv_type::InputBackprop> {
  using ConstPointer =
      typename Backend::template internal_pointer_type<T const>;
  using Pointer = typename Backend::template internal_pointer_type<T>;
  using InternalPointer =
      ::sycldnn::internal::helpers::InternalPointer<T, Backend>;

  WorkspacePointerSet(InternalPointerSet<T, Backend> const& set,
                      typename Backend::template pointer_type<T> workspace,
                      size_t size_per_image, Conv2DParams const& params,
                      size_t workspace_size, Backend& backend)
      : minibatch_size{get_minibatch_size(
            workspace_size,
            filter_transform_size<conv_type::InputBackprop>(params),
            size_per_image)},
        input{set.input.get()},
        original_filter{set.filter.get()},
        filter{workspace, backend},
        transform{
            workspace + filter_transform_size<conv_type::InputBackprop>(params),
            backend},
        output{set.output.get()} {}

  FullPointerSet<T, Backend, conv_type::InputBackprop> to_full_pointer_set() {
    return {input, original_filter, filter.get(), transform.get(), output};
  }

  size_t minibatch_size;
  ConstPointer input;
  ConstPointer original_filter;
  InternalPointer filter;
  InternalPointer transform;
  Pointer output;

 private:
  /** Get the size of minibatch to use for the given workspace size. */
  static size_t get_minibatch_size(size_t workspace_size, size_t filter_size,
                                   size_t size_per_image) {
    size_t transform_workspace_size = workspace_size - filter_size;
    return transform_workspace_size / size_per_image;
  }
};

}  // namespace im2col
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_WORKSPACE_POINTER_SET_H_
