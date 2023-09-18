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
#ifndef PORTDNN_INCLUDE_ROI_ALIGN_PARAMS_H_
#define PORTDNN_INCLUDE_ROI_ALIGN_PARAMS_H_

#include "portdnn/data_format.h"

/**
 * \file
 * Defines the \ref sycldnn::roi_align::RoiAlignParams struct,
 * which contains the values used in a ROI Align operation.
 */
namespace sycldnn {
namespace roi_align {

/**
 * The coordinate transformation mode to use. Used to specify whether to offset
 * the input coordinates.
 */
enum class CoordinateTransformationMode {
  /** Shift the input coordinates by -0.5. */
  HALF_PIXEL,

  /** Do not shift the input coordinates. */
  OUTPUT_HALF_PIXEL,
};

/** Struct that contains values used in a RoiAlign op. */
struct RoiAlignParams {
  /** The type of the params is int, providing a decent
   * upper bound on the tensor sizes.*/
  using Index = int;

  /** The number of tensors in the calculation. */
  Index batch;

  /** The number of channels in each tensor. */
  Index channels;

  /** Height dimensions of the input. */
  Index in_height;

  /** Width dimensions of the input. */
  Index in_width;

  /** Height dimensions of the output. */
  Index out_height = 1;

  /** Width dimensions of the input. */
  Index out_width = 1;

  /**
   * The number of bins over height and width to use to calculate each output
   * feature map element. If set to 0 then an adaptive number of
   * elements over height and width is used: ceil(roi_height / out_h) and
   * ceil(roi_width / out_w) respectively. */
  Index sampling_ratio = 0;

  /**
   * Multiplicative scale factor to translate ROI coordinates from their input
   * spatial scale to the scale used when pooling. */
  float spatial_scale = 1.f;

  /** The number of boxes, i.e. the first dimension of the `rois` input */
  Index num_rois;

  /** The size of the second dimension of the `batch_indices` input */
  const Index roi_cols = 4;

  /** The coordinate transformation mode to use. See
   * \ref CoordinateTransformationMode */
  CoordinateTransformationMode coordinate_transformation_mode =
      CoordinateTransformationMode::OUTPUT_HALF_PIXEL;

  /** The data format used in the input and output tensors. Currently only NCHW
   * is supported. */
  sycldnn::DataFormat input_format = sycldnn::DataFormat::NCHW;
};

}  // namespace roi_align
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_ROI_ALIGN_PARAMS_H_
