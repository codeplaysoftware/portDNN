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
#ifndef PORTDNN_INCLUDE_CONV2D_CONSTANT_SELECTOR_H_
#define PORTDNN_INCLUDE_CONV2D_CONSTANT_SELECTOR_H_

/**
 * \file
 * Contains the definition of the \ref sycldnn::conv2d::ConstantSelector class.
 * This concrete implementations of \ref sycldnn::conv2d::Selector will always
 * select a specific convolution algorithm, regardless of the convolution
 * parameters.
 */
#include "portdnn/conv2d/algorithm.h"
#include "portdnn/conv2d/params.h"
#include "portdnn/conv2d/selector/selector.h"
#include "portdnn/helpers/macros.h"

namespace sycldnn {
namespace conv2d {
/**
 * A selector which will always return the same algorithm, regardless of the
 * convolution parameters passed to the select function.
 */
template <Algorithm Algo>
class ConstantSelector final : public Selector {
 public:
  /**
   * Selects an appropriate convolution algorithm for the target platform, given
   * a set of convolution parameters.
   * \param params The convolution parameters (i.e. the shapes of the tensors,
   *               and strides used by the convolution).
   * \return Returns an instance of \ref sycldnn::conv2d::Algorithm, indicating
   *         the optimal choice of convolution of algorithm.
   */
  Algorithm select_forward(Conv2DParams const& params) override {
    SNN_UNUSED_VAR(params)
    return Algo;
  }

  /**
   * Selects an appropriate convolution algorithm for the target platform, given
   * a set of convolution parameters.
   * \param params The convolution parameters (i.e. the shapes of the tensors,
   *               and strides used by the convolution).
   * \return Returns an instance of \ref sycldnn::conv2d::Algorithm, indicating
   *         the optimal choice of convolution of algorithm.
   */
  Algorithm select_input_backprop(Conv2DParams const& params) override {
    SNN_UNUSED_VAR(params)
    return Algo;
  }

  /**
   * Selects an appropriate convolution algorithm for the target platform, given
   * a set of convolution parameters.
   * \param params The convolution parameters (i.e. the shapes of the tensors,
   *               and strides used by the convolution).
   * \return Returns an instance of \ref sycldnn::conv2d::Algorithm, indicating
   *         the optimal choice of convolution of algorithm.
   */
  Algorithm select_filter_backprop(Conv2DParams const& params) override {
    SNN_UNUSED_VAR(params)
    return Algo;
  }

  /**
   * Gets the name of the selector.
   * \return Returns a character string containing the descriptive name of the
   * selector.
   */
  char const* name() const override {
    switch (Algo) {
      case Algorithm::Direct:
        return "Direct";
      case Algorithm::Tiled:
        return "Tiled";
      case Algorithm::Im2col:
        return "Im2col";
      default:
        SNN_ASSERT(false, "Unsupported algorithm in ConstantSelector::name()");
        return nullptr;
    }
  }
};
}  // namespace conv2d
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_CONV2D_CONSTANT_SELECTOR_H_
