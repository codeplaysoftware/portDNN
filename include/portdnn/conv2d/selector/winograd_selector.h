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
#ifndef PORTDNN_INCLUDE_CONV2D_SELECTOR_WINOGRAD_SELECTOR_H_
#define PORTDNN_INCLUDE_CONV2D_SELECTOR_WINOGRAD_SELECTOR_H_

/**
 * \file
 * Contains the definition of the \ref sycldnn::conv2d::WinogradSelector class.
 * This concrete implementation of \ref sycldnn::conv2d::Selector will always
 * attempt to select the Winograd convolution algorithm when supported.
 */
#include "portdnn/conv2d/algorithm.h"
#include "portdnn/conv2d/params.h"

#include "portdnn/conv2d/selector/selector.h"

namespace sycldnn {
namespace conv2d {

/** A selector which returns the Winograd algorithm if supported. */
class WinogradSelector final : public Selector {
 public:
  /**
   * Selects an appropriate convolution algorithm for the target platform, given
   * a set of convolution parameters for forward convolutions.
   * \param params The convolution parameters (i.e. the shapes of the tensors,
   * and strides used by the convolution).
   * \return Returns Algorithm::Winograd when the Winograd algorithm is
   * supported, or Algorithm::NotSupported otherwise.
   */
  Algorithm select_forward(Conv2DParams const& params) override {
    if (params.stride_rows != 1 && params.stride_cols != 1) {
      return Algorithm::NotSupported;
    }
    if (params.window_rows == 1 && params.window_cols == 3) {
      return Algorithm::Winograd;
    }
    if (params.window_rows == 3 && params.window_cols == 1) {
      return Algorithm::Winograd;
    }
    if (params.window_rows == 3 && params.window_cols == 3) {
      return Algorithm::Winograd;
    }
    return Algorithm::NotSupported;
  }

  /**
   * Selects an appropriate convolution algorithm for the target platform, given
   * a set of convolution parameters for input backprop convolutions.
   * \param params The convolution parameters (i.e. the shapes of the tensors,
   * and strides used by the convolution).
   * \return Returns Algorithm::Winograd when the Winograd algorithm is
   * supported, or Algorithm::NotSupported otherwise.
   */
  Algorithm select_input_backprop(Conv2DParams const& params) override {
    if (params.stride_rows != 1 && params.stride_cols != 1) {
      return Algorithm::NotSupported;
    }
    if (params.window_rows == 1 && params.window_cols == 3) {
      return Algorithm::Winograd;
    }
    if (params.window_rows == 3 && params.window_cols == 1) {
      return Algorithm::Winograd;
    }
    if (params.window_rows == 3 && params.window_cols == 3) {
      return Algorithm::Winograd;
    }
    return Algorithm::NotSupported;
  }

  /**
   * Selects an appropriate convolution algorithm for the target platform, given
   * a set of convolution parameters for filter backprop convolutions.
   * \param params The convolution parameters (i.e. the shapes of the tensors,
   * and strides used by the convolution).
   * \return Returns Algorithm::Winograd when the Winograd algorithm is
   * supported, or Algorithm::NotSupported otherwise.
   */
  Algorithm select_filter_backprop(Conv2DParams const& params) override {
    if (params.stride_rows != 1 && params.stride_cols != 1) {
      return Algorithm::NotSupported;
    }
    if (params.window_rows == 1 && params.window_cols == 3) {
      return Algorithm::Winograd;
    }
    if (params.window_rows == 3 && params.window_cols == 1) {
      return Algorithm::Winograd;
    }
    if (params.window_rows == 3 && params.window_cols == 3) {
      return Algorithm::Winograd;
    }
    return Algorithm::NotSupported;
  }

  /**
   * Gets the name of the selector.
   * \return Returns a character string containing the descriptive name of the
   * selector.
   */
  char const* name() const override { return "WinogradSelector"; }
};

/** A selector which returns the WinogradLarge algorithm if supported. */
class WinogradLargeSelector final : public Selector {
 public:
  /**
   * Selects the WinogradLarge algorithm when supported for the provided
   * convolution parameters, otherwise NotSupported.
   *
   * \param params The convolution parameters (i.e. the shapes of the tensors,
   * and strides used by the convolution).
   * \return Returns Algorithm::WinogradLarge when the Winograd algorithm is
   * supported, or Algorithm::NotSupported otherwise.
   */
  Algorithm select_forward(Conv2DParams const& params) override {
    if (params.stride_rows != 1 && params.stride_cols != 1) {
      return Algorithm::NotSupported;
    }
    if (params.window_rows == 3 && params.window_cols == 3) {
      return Algorithm::WinogradLarge;
    }
    return Algorithm::NotSupported;
  }

  /**
   * Selects the WinogradLarge algorithm when supported for the provided
   * convolution parameters, otherwise NotSupported.
   *
   * \param params The convolution parameters (i.e. the shapes of the tensors,
   * and strides used by the convolution).
   * \return Returns Algorithm::WinogradLarge when the Winograd algorithm is
   * supported, or Algorithm::NotSupported otherwise.
   */
  Algorithm select_input_backprop(Conv2DParams const& params) override {
    if (params.stride_rows != 1 && params.stride_cols != 1) {
      return Algorithm::NotSupported;
    }
    if (params.window_rows == 3 && params.window_cols == 3) {
      return Algorithm::WinogradLarge;
    }
    return Algorithm::NotSupported;
  }

  /**
   * Selects the WinogradLarge algorithm when supported for the provided
   * convolution parameters, otherwise NotSupported.
   *
   * \param params The convolution parameters (i.e. the shapes of the tensors,
   * and strides used by the convolution).
   * \return Returns Algorithm::WinogradLarge when the Winograd algorithm is
   * supported, or Algorithm::NotSupported otherwise.
   */
  Algorithm select_filter_backprop(Conv2DParams const& params) override {
    if (params.stride_rows != 1 && params.stride_cols != 1) {
      return Algorithm::NotSupported;
    }
    if (params.window_rows == 3 && params.window_cols == 3) {
      return Algorithm::WinogradLarge;
    }
    return Algorithm::NotSupported;
  }

  /**
   * Gets the name of the selector.
   * \return Returns a character string containing the descriptive name of the
   * selector.
   */
  char const* name() const override { return "WinogradLargeSelector"; }
};

}  // namespace conv2d
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_CONV2D_SELECTOR_WINOGRAD_SELECTOR_H_
