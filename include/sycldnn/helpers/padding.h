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
#ifndef SYCLDNN_INCLUDE_HELPERS_PADDING_H_
#define SYCLDNN_INCLUDE_HELPERS_PADDING_H_

/**
 * \file
 * Contains helper functions to compute required output sizes and padding, based
 * on convolution strides and filter sizes.
 */
#include "sycldnn/padding_mode.h"

#include "sycldnn/helpers/ratio.h"

namespace sycldnn {
namespace helpers {
/** A simple struct for padding and output sizes. */
template <typename Index>
struct PaddingAndOutput {
  /** The number of required padding elements. */
  Index padding;

  /** The number of output elements. */
  Index output;
};
/**
 * Calculate the padding and output size given the input size, window and
 * stride.
 */
template <typename Index>
PaddingAndOutput<Index> calculate_padding(Index input, Index window,
                                          Index stride, PaddingMode type) {
  switch (type) {
    case PaddingMode::VALID: {
      Index output = round_ratio_up(input - window + 1, stride);
      Index padding = 0;
      return PaddingAndOutput<Index>{padding, output};
    }
    case PaddingMode::SAME: {
      Index output = round_ratio_up(input, stride);
      Index padding_needed = (output - 1) * stride + window - input;
      Index padding = padding_needed / 2;
      return PaddingAndOutput<Index>{padding, output};
    }
    default:
      SNN_ASSERT(false, "Invalid padding mode passed to calculate_padding");
      return PaddingAndOutput<Index>{0, 0};
  }
}
}  // namespace helpers
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_HELPERS_PADDING_H_
