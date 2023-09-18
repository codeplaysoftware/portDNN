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
#ifndef PORTDNN_INCLUDE_CONV2D_DIRECT_SELECTOR_H_
#define PORTDNN_INCLUDE_CONV2D_DIRECT_SELECTOR_H_

/**
 * \file
 * Contains the definition of the \ref sycldnn::conv2d::DirectSelector alias.
 * This \ref sycldnn::conv2d::Selector will always select the direct convolution
 * algorithm, regardless of the convolution parameters.
 */
#include "portdnn/conv2d/selector/constant_selector.h"

namespace sycldnn {
namespace conv2d {
/** A selector which always returns the Direct algorithm. */
using DirectSelector = ConstantSelector<Algorithm::Direct>;
}  // namespace conv2d
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_CONV2D_DIRECT_SELECTOR_H_
