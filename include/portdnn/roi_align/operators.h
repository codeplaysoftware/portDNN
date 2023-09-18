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

#ifndef PORTDNN_INCLUDE_ROI_ALIGN_OPERATORS_H_
#define PORTDNN_INCLUDE_ROI_ALIGN_OPERATORS_H_

#include "portdnn/pooling/operators.h"

namespace sycldnn {
namespace roi_align {

/** Type alias for the Max pooling operator. */
template <typename T>
struct MaxPool : public sycldnn::pooling::Max<T> {};

/** Type alias for the Average pooling operator.  */
template <typename T>
struct AveragePool : public sycldnn::pooling::Average<T> {};

}  // namespace roi_align
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_ROI_ALIGN_OPERATORS_H_
