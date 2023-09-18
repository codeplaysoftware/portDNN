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
#ifndef PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_OFFSETS_H_
#define PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_OFFSETS_H_

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/params.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace im2col {

/** Offsets for a given minibatch. */
struct Offsets {
  /** Offset into the input tensor. */
  size_t in;
  /** Offset into the output tensor. */
  size_t out;
};

/**
 * Calculate the offsets into the input and output tensors at mini-batch `i` for
 * mini-batches of size `minibatch_size`.
 *
 * \param i              Index of which minibatch to calculate offsets for
 * \param minibatch_size Number of images in a minibatch
 * \param params         User provided conv2d parameters
 * \return An Offsets struct containing input and output offsets
 */
template <typename ConvType>
inline Offsets calculate_offsets(size_t i, size_t minibatch_size,
                                 Conv2DParams const& params) {
  const size_t in_offset =
      i * minibatch_size * params.in_rows * params.in_cols * params.channels;
  const size_t out_offset =
      i * minibatch_size * params.out_rows * params.out_cols * params.features;
  return Offsets{in_offset, out_offset};
}

template <>
inline Offsets calculate_offsets<conv_type::InputBackprop>(
    size_t i, size_t minibatch_size, Conv2DParams const& params) {
  const size_t in_offset =
      i * minibatch_size * params.out_rows * params.out_cols * params.features;
  const size_t out_offset =
      i * minibatch_size * params.in_rows * params.in_cols * params.channels;
  return Offsets{in_offset, out_offset};
}

}  // namespace im2col
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_OFFSETS_H_
