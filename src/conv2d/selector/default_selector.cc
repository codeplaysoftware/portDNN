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
#include "sycldnn/conv2d/selector/default_selector.h"
#include "sycldnn/conv2d/selector/direct_selector.h"

namespace {

/**
 * Implements a reasonable default selector. This currently just returns the
 * direct convolution algorithm, but provides a location to extend with greater
 * intelligence in future.
 */
class DefaultSelector final : public sycldnn::conv2d::Selector {
 public:
  /**
   * Selects an appropriate convolution algorithm for the target platform, given
   * a set of convolution parameters.
   * \param params The convolution parameters (i.e. the shapes of the tensors,
   *               and strides used by the convolution).
   * \return Returns an instance of \ref sycldnn::conv2d::Algorithm, indicating
   *         the optimal choice of convolution of algorithm.
   */
  sycldnn::conv2d::Algorithm select(
      sycldnn::conv2d::Conv2DParams const& params) override {
    SNN_UNUSED_VAR(params)
    return sycldnn::conv2d::Algorithm::Direct;
  }

  /**
   * Gets the name of the selector.
   * \return Returns a character string containing the descriptive name of the
   * selector.
   */
  char const* name() const override { return "Default"; }
};
}

namespace sycldnn {
namespace conv2d {

std::unique_ptr<Selector> get_default_selector(const cl::sycl::device&) {
  return std::unique_ptr<Selector>{
      static_cast<Selector*>(new DefaultSelector{})};
}

}  // namespace conv2d
}  // namespace sycldnn
