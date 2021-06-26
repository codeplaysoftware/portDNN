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
#ifndef SYCLDNN_INCLUDE_BIAS_PARAMS_H_
#define SYCLDNN_INCLUDE_BIAS_PARAMS_H_

#include "sycldnn/data_format.h"

/**
 * \file
 * Defines the \ref sycldnn::bias::BiasParams struct,
 * which contains the values used in a bias operation.
 */
namespace sycldnn {
namespace bias {

/** Struct that contains values used in a Bias op. */
struct BiasParams {
  /** The type of the params is int, providing a decent
   * upper bound on the tensor sizes.*/
  using Index = int;

  /** The number of input rows. */
  Index in_rows;

  /** The number of input columns. */
  Index in_cols;

  /** The number of tensors in the calculation. */
  Index batch;

  /** The number of channels in each tensor. */
  Index channels;

  /** The number of bias values*/
  Index bias;

  /** The data format used in the input and output tensors. */
  sycldnn::DataFormat input_format = sycldnn::DataFormat::NHWC;
};

}  // namespace bias
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_BIAS_PARAMS_H_
