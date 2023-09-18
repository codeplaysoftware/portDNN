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
#ifndef PORTDNN_INCLUDE_INTERNAL_CONV2D_WINOGRAD_CALCULATE_OFFSETS_H_
#define PORTDNN_INCLUDE_INTERNAL_CONV2D_WINOGRAD_CALCULATE_OFFSETS_H_

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/params.h"

/**
 * \file
 * Contains the definition of \ref sycldnn::conv2d::internal::winograd::Offsets
 * struct and corresponding helper function
 * sycldnn::conv2d::internal::winograd::calculate_offsets() to calculate the
 * offsets into a tensor for a particular minibatch.
 */

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace winograd {

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
inline Offsets calculate_offsets(int i, int minibatch_size,
                                 Conv2DParams const& params) {
  size_t const in_offset =
      i * minibatch_size * params.in_rows * params.in_cols * params.channels;
  size_t const out_offset =
      i * minibatch_size * params.out_rows * params.out_cols * params.features;
  Offsets result{in_offset, out_offset};
  return result;
}

/** \copydoc calculate_offsets() */
template <>
inline Offsets calculate_offsets<conv_type::FilterBackprop>(
    int i, int minibatch_size, Conv2DParams const& params) {
  size_t const in_offset =
      i * minibatch_size * params.in_rows * params.in_cols * params.channels;
  size_t const out_offset = i * minibatch_size * params.window_rows *
                            params.window_cols * params.features;
  Offsets result{in_offset, out_offset};
  return result;
}

}  // namespace winograd
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_INTERNAL_CONV2D_WINOGRAD_CALCULATE_OFFSETS_H_
