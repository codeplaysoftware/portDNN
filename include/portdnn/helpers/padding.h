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
#ifndef PORTDNN_INCLUDE_HELPERS_PADDING_H_
#define PORTDNN_INCLUDE_HELPERS_PADDING_H_

/**
 * \file
 * Contains helper functions to compute required output sizes and padding, based
 * on convolution strides and filter sizes.
 */
#include "portdnn/padding_mode.h"

#include "portdnn/helpers/ratio.h"

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
 * stride. Padding is independent per-dimension.
 * \param input The size of the tensor to be padded.
 * \param window The size of the filter that will be convolved with
 *               the input.
 * \param stride The stride that the window will be advanced by.
 * \param type The type of padding that will be applied.
 * \return The padding and new size of the tensor as a pair (POD struct).
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

/**
 * Add the padding and output sizes to a parameter struct from the input
 * sizes, window sizes and strides.
 * \param params The parameters that the output will be based on.
 * \param type The type of padding that should be used to calculate the actual
 *             size of padding to be used in the convolution.
 * \return The original params, modified with the padding sizes required.
 */
template <typename Params>
Params add_padding_to(Params params, PaddingMode type) {
  auto row_padding = sycldnn::helpers::calculate_padding(
      params.in_rows, params.window_rows, params.stride_rows, type);
  params.out_rows = row_padding.output;
  params.pad_rows = row_padding.padding;

  auto col_padding = sycldnn::helpers::calculate_padding(
      params.in_cols, params.window_cols, params.stride_cols, type);
  params.out_cols = col_padding.output;
  params.pad_cols = col_padding.padding;

  return params;
}

}  // namespace helpers
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_HELPERS_PADDING_H_
