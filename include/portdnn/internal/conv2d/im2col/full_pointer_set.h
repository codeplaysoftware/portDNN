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
#ifndef PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_FULL_POINTER_SET_H_
#define PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_FULL_POINTER_SET_H_

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace im2col {

/**
 * Set of all pointers required for im2col.
 */
template <typename T, typename Backend, typename ConvType>
struct FullPointerSet {
  using ConstPointer =
      typename Backend::template internal_pointer_type<T const>;
  using Pointer = typename Backend::template internal_pointer_type<T>;

  ConstPointer input;
  ConstPointer filter;
  Pointer transform;
  Pointer output;
};

/**
 * Set of all pointers required for im2col input backprop.
 *
 * The input backprop required an additional temporary buffer to hold the filter
 * transform which is not needed in the other cases.
 */
template <typename T, typename Backend>
struct FullPointerSet<T, Backend, conv_type::InputBackprop> {
  using ConstPointer =
      typename Backend::template internal_pointer_type<T const>;
  using Pointer = typename Backend::template internal_pointer_type<T>;

  ConstPointer input;
  ConstPointer original_filter;
  Pointer filter;
  Pointer transform;
  Pointer output;
};

}  // namespace im2col
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_FULL_POINTER_SET_H_
