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
#ifndef PORTDNN_INCLUDE_PADDING_MODE_H_
#define PORTDNN_INCLUDE_PADDING_MODE_H_

/**
 * \file
 * Contains the declaration of the \ref sycldnn::PaddingMode enumerated type.
 * This type is used to control boundary mode behaviour and padding for
 * convolutions.
 */
namespace sycldnn {
/**
 * Padding Mode to use, similar to how Eigen specifies its padding.
 *
 * The padding and output sizes will differ depending on which PaddingMode is
 * used. These follow the padding modes specified by Eigen:
 *
 *  - PaddingMode::VALID will provide an output sized such that no padding is
 *    used on the input.
 *
 *  - PaddingMode::SAME will provide an output sized to match the input if the
 *    stride is 1, with padding added to ensure that the output will be that
 *    size. If the stride is larger than 1 then the padding will be the same as
 *    for stride 1 and the output size will be decreased to match the stride.
 */
enum class PaddingMode { VALID, SAME };

}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_HELPERS_PADDING_H_
