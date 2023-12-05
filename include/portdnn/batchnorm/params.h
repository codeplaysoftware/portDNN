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
#ifndef PORTDNN_INCLUDE_BATCHNORM_PARAMS_H_
#define PORTDNN_INCLUDE_BATCHNORM_PARAMS_H_

#include "portdnn/data_format.h"

/**
 * \file
 * Contains the declaration of the \ref sycldnn::batchnorm::BatchNormParams
 * structure, which represents the tensor shapes for a batchnorm operation.
 */
namespace sycldnn {
namespace batchnorm {

/** Parameter struct containing the parameters required for a batchnorm
 * operation.
 */
struct BatchNormParams {
  /** The underlying data type of all index parameters. */
  using Index = int;

  /** The number of input/output tensors per batch. */
  Index batch;

  /** The number of rows in each input/output tensor. */
  Index rows;

  /** The number of columns in each input/output tensor. */
  Index cols;

  /**
   * The number of channels (or feature maps) in each input/output tensor.
   * Can also be read as the number of classes in a classification task.
   */
  Index channels;

  /**
   * Set to true to use Batchnorm in training mode.
   * In training mode the mean and variance are computed on the go.
   * Set to false to use Batchnorm in frozen mode.
   * In frozen mode the mean and variance are provided as inputs.
   */
  bool is_training;

  /**
   * The epsilon parameter for batch normalization to ensure divisibility by a
   * non-zero value.
   */
  float epsilon = 0.001;

  /**
   * The momentum parameter for batch normalization.
   */
  float momentum = 0.9;

  /** The data format used in the input and output tensors. */
  sycldnn::DataFormat input_format = sycldnn::DataFormat::NHWC;
};

}  // namespace batchnorm
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_BATCHNORM_PARAMS_H_