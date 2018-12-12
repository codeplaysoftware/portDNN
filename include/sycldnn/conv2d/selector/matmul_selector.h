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
#ifndef SYCLDNN_INCLUDE_CONV2D_SELECTOR_MATMUL_SELECTOR_H_
#define SYCLDNN_INCLUDE_CONV2D_SELECTOR_MATMUL_SELECTOR_H_

/**
 * \file
 * Contains the definition of the \ref sycldnn::conv2d::MatmulSelector class.
 * This concrete implementation of \ref sycldnn::conv2d::Selector will always
 * attempt to select the matmul convolution algorithm when supported.
 */
#include "sycldnn/conv2d/algorithm.h"
#include "sycldnn/conv2d/params.h"

#include "sycldnn/conv2d/selector/selector.h"

namespace sycldnn {
namespace conv2d {

/** A selector which returns the matmul algorithm if supported. */
class MatmulSelector final : public Selector {
 public:
  /**
   * Selects an appropriate convolution algorithm for the target platform, given
   * a set of convolution parameters.
   * \param params The convolution parameters (i.e. the shapes of the tensors,
   * and strides used by the convolution).
   * \return Returns Algorithm::Matmul when the matmul algorithm is
   * supported, or Algorithm::NotSupported otherwise.
   */
  Algorithm select(Conv2DParams const& params) override {
    // TODO(jwlawson): Ensure the data format is NHWC
    bool right_stride = (params.stride_rows == 1 && params.stride_cols == 1);
    bool right_window = (params.window_rows == 1 && params.window_cols == 1);
    bool right_pad = (params.pad_rows == 0 && params.pad_cols == 0);

    if (right_stride && right_window && right_pad) {
      return Algorithm::Matmul;
    } else {
      return Algorithm::NotSupported;
    }
  }

  /**
   * Gets the name of the selector.
   * \return Returns a character string containing the descriptive name of the
   * selector.
   */
  char const* name() const override { return "MatmulSelector"; }
};

}  // namespace conv2d
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_CONV2D_SELECTOR_MATMUL_SELECTOR_H_
