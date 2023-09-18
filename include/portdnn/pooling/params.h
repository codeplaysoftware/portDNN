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
 * distributed under the License is distributed on an "AS IS" BASIS
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef PORTDNN_INCLUDE_POOLING_PARAMS_H_
#define PORTDNN_INCLUDE_POOLING_PARAMS_H_

#include "portdnn/data_format.h"

/**
 * \file
 * Defines the \ref sycldnn::pooling::PoolingParams struct,
 * which contains the values used in a pooling operation.
 */
namespace sycldnn {
namespace pooling {

/** Struct that contains values used in a Pooling op. */
struct PoolingParams {
  /** The type of the params is int, providing a decent
   * upper bound on the tensor sizes.*/
  using Index = int;

  /** The number of input rows. */
  Index in_rows;

  /** The number of input columns. */
  Index in_cols;

  /** The number of output rows. */
  Index out_rows;

  /** The number of output columns. */
  Index out_cols;

  /** The number of pooling window rows. */
  Index window_rows;

  /** The number of pooling window columns. */
  Index window_cols;

  /** The stride of the window down the rows. */
  Index stride_rows;

  /** The stride of the window along the columns. */
  Index stride_cols;

  /** The number of tensors in the calculation. */
  Index batch;

  /** The number of channels in each tensor. */
  Index channels;

  /** The padding to be applied to each tensor row. */
  Index pad_rows;

  /** The padding to be applied to each tensor column. */
  Index pad_cols;

  /** The data format used in the input and output tensors. */
  sycldnn::DataFormat input_format = sycldnn::DataFormat::NHWC;
};

}  // namespace pooling
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_POOLING_PARAMS_H_
