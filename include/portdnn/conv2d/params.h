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
#ifndef PORTDNN_INCLUDE_CONV2D_PARAMS_H_
#define PORTDNN_INCLUDE_CONV2D_PARAMS_H_

#include "portdnn/batch_format.h"
#include "portdnn/data_format.h"
#include "portdnn/filter_format.h"

/**
 * \file
 * Contains the declaration of the \ref sycldnn::conv2d::Conv2DParams structure,
 * which represents the tensor shapes and convolution strides for a 2D
 * convolution.
 */
namespace sycldnn {
namespace conv2d {
/** Parameter struct containing the parameters required for a 2D convolution. */
struct Conv2DParams {
  /** The underlying data type of all index parameters. */
  using Index = int;

  /** The number of channels (or feature maps) in each input image. */
  Index channels;

  /** The number of feature maps (or channels) in each output image. */
  Index features;

  /** The number of input images per batch. */
  Index batch;

  /** The number of rows in each input image. */
  Index in_rows;

  /** The number of columns in each input image. */
  Index in_cols;

  /** The number of rows in the filter kernel (or convolution matrix). */
  Index window_rows;

  /** The number of columns in the filter kernel (or convolution matrix). */
  Index window_cols;

  /**
   * The number of elements within the input image to horizontally shift the
   * filter kernel per output sample.
   */
  Index stride_rows;

  /**
   * The number of elements within the input image to vertically shift the
   * filter kernel per output sample.
   */
  Index stride_cols;

  /** The number of rows in each output image. */
  Index out_rows;

  /** The number of columns in each output image. */
  Index out_cols;

  /**
   * The number of rows to zero-pad the input images by.
   */
  Index pad_rows;

  /**
   * The number of columns to zero-pad the input images by.
   */
  Index pad_cols;

  /**
   * Horizontal stride between elements sampled in the input image. Given a
   * filter w, and an input image x, a 1D dense convolution might compute:
   * w[0] * x[0] + w[1] * x[1] + w[2] * x[2]. By contrast, a dilated convolution
   * might compute: w[0] * x[0] + w[1] * x[2] + w[2] * x[4].
   */
  Index dilation_rows = 1;

  /**
   * Vertical stride between elements sampled in the input image.
   */
  Index dilation_cols = 1;

  /**
   * Number of feature map groups for grouped convolution.
   */
  Index groups = 1;

  /** The data format used in the input and output tensors. */
  sycldnn::DataFormat input_format = sycldnn::DataFormat::NHWC;

  /** The data format used in the filter tensor. */
  sycldnn::FilterFormat filter_format = sycldnn::FilterFormat::HWCF;

  /** The data format used in the input tensor to specify how the groups are
   * laid out*/
  sycldnn::BatchFormat group_format = sycldnn::BatchFormat::STRIDED;
};
}  // namespace conv2d
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_CONV2D_PARAMS_H_
