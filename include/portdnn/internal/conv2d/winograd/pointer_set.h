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
#ifndef PORTDNN_INCLUDE_INTERNAL_CONV2D_WINOGRAD_POINTER_SET_H_
#define PORTDNN_INCLUDE_INTERNAL_CONV2D_WINOGRAD_POINTER_SET_H_

#include "portdnn/conv2d/params.h"

#include "portdnn/internal/conv2d/alloc_info.h"
#include "portdnn/internal/conv2d/internal_pointer_set.h"

#include "portdnn/internal/helpers/allocated_pointer.h"

#include "portdnn/internal/conv2d/winograd/tile_info.h"

/**
 * \file
 * Contains the sycldnn::conv2d::internal::winograd::FullPointerSet to wrap the
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

}  // namespace winograd
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_INTERNAL_CONV2D_WINOGRAD_POINTER_SET_H_
