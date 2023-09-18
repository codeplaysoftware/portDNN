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
#ifndef PORTDNN_INCLUDE_CONV2D_SELECTOR_H_
#define PORTDNN_INCLUDE_CONV2D_SELECTOR_H_

/**
 * \file
 * Contains the definition of the \ref sycldnn::conv2d::Selector abstract base
 * class. Concrete implementations of \ref sycldnn::conv2d::Selector enable
 * portDNN to select the most appropriate convolution algorithm for a specific
 * target platform or scenario.
 */
#include "portdnn/conv2d/algorithm.h"
#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/params.h"

namespace sycldnn {
namespace conv2d {
/**
 * Class to select which convolution implementation to use for a given set of
 * parameters. Can be implemented for different devices which exhibit different
 * performance characteristics.
 */
class Selector {
 public:
  /**
   * Selects an appropriate convolution algorithm for the target platform, given
   * a set of convolution parameters.
   * \param params The convolution parameters (i.e. the shapes of the tensors,
   * and strides used by the convolution).
   * \return Returns an instance of \ref sycldnn::conv2d::Algorithm, indicating
   *  the optimal choice of convolution of algorithm.
   */
  template <typename ConvType>
  Algorithm select(Conv2DParams const& params);

  /**
   * Overrideable function that selects algorithms for forward convolutions.
   * \param params The convolution parameters.
   * \return Returns a \ref sycldnn::conv2d::Algorithm.
   */
  virtual Algorithm select_forward(Conv2DParams const& params) = 0;

  /**
   * Overrideable function that selects algorithms for input backprop
   * convolutions.
   * \param params The convolution parameters.
   * \return Returns a
   * \ref sycldnn::conv2d::Algorithm.
   */
  virtual Algorithm select_input_backprop(Conv2DParams const& params) = 0;

  /**
   * Overrideable function that selects algorithms for filter backprop
   * convolutions.
   * \param params The convolution parameters.
   * \return Returns a
   * \ref sycldnn::conv2d::Algorithm.
   */
  virtual Algorithm select_filter_backprop(Conv2DParams const& params) = 0;

  /**
   * Gets the name of the selector.
   * \return Returns a character string containing the descriptive name of the
   * selector.
   */
  virtual char const* name() const = 0;

  /**
   * Virtual destructor to ensure that the destructor of a derived class is
   * called.
   */
  virtual ~Selector() = default;
};

/** \copydoc Selector::select() */
template <>
inline Algorithm Selector::select<conv_type::Forward>(
    Conv2DParams const& params) {
  return this->select_forward(params);
}

/** \copydoc Selector::select() */
template <>
inline Algorithm Selector::select<conv_type::InputBackprop>(
    Conv2DParams const& params) {
  return this->select_input_backprop(params);
}

/** \copydoc Selector::select() */
template <>
inline Algorithm Selector::select<conv_type::FilterBackprop>(
    Conv2DParams const& params) {
  return this->select_filter_backprop(params);
}
}  // namespace conv2d
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_CONV2D_SELECTOR_H_
