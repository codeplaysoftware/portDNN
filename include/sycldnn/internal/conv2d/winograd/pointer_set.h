/*
 * Copyright 2018 Codeplay Software Ltd
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
#ifndef SYCLDNN_INCLUDE_INTERNAL_CONV2D_WINOGRAD_POINTER_SET_H_
#define SYCLDNN_INCLUDE_INTERNAL_CONV2D_WINOGRAD_POINTER_SET_H_

#include "sycldnn/conv2d/params.h"

#include "sycldnn/internal/conv2d/alloc_info.h"
#include "sycldnn/internal/conv2d/internal_pointer_set.h"

#include "sycldnn/internal/helpers/allocated_pointer.h"

#include "sycldnn/internal/conv2d/winograd/tile_info.h"

/**
 * \file
 * Contains the sycldnn::conv2d::internal::winograd::FullPointerSet and
 * sycldnn::conv2d::internal::winograd::AllocatedPointerSet helpers to wrap the
 * pointers required for a Winograd convolution.
 */

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace winograd {

/**
 * Struct containing all the pointers required for a Winograd convolution,
 * including the user provided pointers and the pointers to the temporary
 * transform buffers.
 */
template <typename T, typename Backend>
struct FullPointerSet {
  /** User provided internal pointer type. */
  using Pointer = typename Backend::template internal_pointer_type<T>;
  /** User provided internal const pointer type. */
  using ConstPointer =
      typename Backend::template internal_pointer_type<T const>;

  FullPointerSet() = delete;

  /** The user provided input pointer. */
  ConstPointer input;
  /** The user provided filter pointer. */
  ConstPointer filter;
  /** The user provided output pointer. */
  Pointer output;
  /** The temporary input transform pointer. */
  Pointer input_transform;
  /** The temporary filter transform pointer. */
  Pointer filter_transform;
  /** The temporary output transform pointer. */
  Pointer intermediate;
};

/**
 * Container to allocate the temporary buffers required for a Winograd
 * convolution.
 *
 * On construction, this will use the provided backend to allocate the temporary
 * buffers required to hold the transformed data used in the convolution. On
 * destruction, these pointers will be released through the backend.
 *
 * The temporary buffers will be allocated large enough to hold a specific
 * minibatch size, which is computed to be as large as possible without causing
 * the SYCL device to fail to allocate memory.
 */
template <typename T, typename Backend>
struct AllocatedPointerSet {
  /** User provided internal pointer type. */
  using Pointer = typename Backend::template internal_pointer_type<T>;
  /** User provided internal const pointer type. */
  using ConstPointer =
      typename Backend::template internal_pointer_type<T const>;
  /** RAII allocating pointer which releases its buffer on destruction. */
  using AllocatedPointer =
      ::sycldnn::internal::helpers::AllocatedPointer<T, Backend>;

  /**
   * Construct an AllocatedPointerSet from the user provided pointers and the
   * convolution parameters. Will allocate the temporary buffers required
   * through the backend.
   *
   * \param set        The user provided pointers.
   * \param params     The parameters for the convolution.
   * \param n_matrices The number of matrices in the intermediate Winograd
   *                   tensor.
   * \param tile_info  Information about the number of tiles in the convolution.
   * \param backend    The user provided backend to use to allocate the
   *                   temporary buffers.
   */
  AllocatedPointerSet(InternalPointerSet<T, Backend> const& set,
                      Conv2DParams const& params, int n_matrices,
                      TileInfo const& tile_info, Backend& backend)
      : minibatch_size{get_minibatch_size(params, tile_info, n_matrices,
                                          backend)},
        input{set.input.get()},
        filter{set.filter.get()},
        output{set.output.get()},
        input_transform{minibatch_size * in_transform_bytes_per_image(
                                             params, tile_info, n_matrices),
                        backend},
        filter_transform{filter_transform_bytes(params, n_matrices), backend},
        intermediate{minibatch_size * intermediate_bytes_per_image(
                                          params, tile_info, n_matrices),
                     backend} {}

  /**
   * Convert an AllocatedPointerSet to a FullPointerSet instance.
   * \return A FullPointerSet containing the pointers in this
   * AllocatedPointerSet.
   */
  FullPointerSet<T, Backend> to_full_pointer_set() const {
    return {input,
            filter,
            output,
            input_transform.get(),
            filter_transform.get(),
            intermediate.get()};
  }

  /** Minibatch size used to allocate the temporary buffers. */
  size_t minibatch_size;
  /** The user provided input pointer. */
  ConstPointer input;
  /** The user provided filter pointer. */
  ConstPointer filter;
  /** The user provided output pointer. */
  Pointer output;
  /** The temporary input transform pointer. */
  AllocatedPointer input_transform;
  /** The temporary filter transform pointer. */
  AllocatedPointer filter_transform;
  /** The temporary output transform pointer. */
  AllocatedPointer intermediate;

 private:
  /** Number of bytes per image required in the input transform tensor. */
  static size_t in_transform_bytes_per_image(Conv2DParams const& params,
                                             TileInfo const& tile_info,
                                             int n_matrices) {
    return sizeof(T) * n_matrices * tile_info.number * params.channels;
  }
  /** Number of bytes per image required in the intermediate tensor. */
  static size_t intermediate_bytes_per_image(Conv2DParams const& params,
                                             TileInfo const& tile_info,
                                             int n_matrices) {
    return sizeof(T) * n_matrices * tile_info.number * params.features;
  }
  /** Number of bytes per image required in the filter transform tensor. */
  static size_t filter_transform_bytes(Conv2DParams const& params,
                                       int n_matrices) {
    return sizeof(T) * n_matrices * params.channels * params.features;
  }
  /** Maximum number of bytes required for the temporary buffers per image. */
  static size_t max_bytes_per_image(Conv2DParams const& params,
                                    TileInfo const& tile_info, int n_matrices) {
    size_t in_transform_bytes =
        in_transform_bytes_per_image(params, tile_info, n_matrices);
    size_t intermediate_bytes =
        intermediate_bytes_per_image(params, tile_info, n_matrices);
    size_t max_bytes_per_image =
        std::max(in_transform_bytes, intermediate_bytes);
    return max_bytes_per_image;
  }
  /** Get the number of images to use per minibatch. */
  static size_t get_minibatch_size(Conv2DParams const& params,
                                   TileInfo const& tile_info, int n_matrices,
                                   Backend& backend) {
    auto max_bytes = max_bytes_per_image(params, tile_info, n_matrices);
    auto alloc_info = get_alloc_info(backend.get_queue().get_device(),
                                     params.batch, max_bytes);
    if (alloc_info.alloc_warning) {
      // TODO: Handle error
    }
    return alloc_info.images_per_alloc;
  }
};

}  // namespace winograd
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_INTERNAL_CONV2D_WINOGRAD_POINTER_SET_H_
