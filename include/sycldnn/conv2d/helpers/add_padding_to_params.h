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
#ifndef SYCLDNN_INCLUDE_CONV2D_HELPERS_ADD_PADDING_TO_PARAMS_H_
#define SYCLDNN_INCLUDE_CONV2D_HELPERS_ADD_PADDING_TO_PARAMS_H_

/**
 * \file
 * Contains helper functions to add the padding and output sizes to a
 * \ref sycldnn::conv2d::Conv2DParams parameter struct from the input sizes,
 * window sizes and strides.
 */
#include "sycldnn/padding_mode.h"

#include "sycldnn/helpers/padding.h"

#include "sycldnn/conv2d/params.h"

namespace sycldnn {
namespace conv2d {
namespace helpers {
/**
 * Add the padding and output sizes to a conv2d parameter struct from the input
 * sizes, window sizes and strides.
 */
conv2d::Conv2DParams add_padding_to(conv2d::Conv2DParams params,
                                    PaddingMode type) {
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
}  // namespace conv2d
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_CONV2D_HELPERS_ADD_PADDING_TO_PARAMS_H_
