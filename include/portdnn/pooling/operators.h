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

#ifndef PORTDNN_INCLUDE_POOLING_OPERATORS_H_
#define PORTDNN_INCLUDE_POOLING_OPERATORS_H_

namespace sycldnn {
namespace pooling {

template <typename T>
struct Max;

/**
 * Max pooling operator which treats NaN as a maximal value.
 *
 * This will force NaNs to propagate through max pooling layers, and so prevents
 * any error in calculation from being silently replaced with a valid value. Any
 * pooling window which contains a NaN will return NaN as the result of the
 * pooling layer, otherwise the result will be the same as with max pooling.
 */
template <typename T>
struct MaxWithNan;

template <typename T>
struct Average;

struct Forward;

struct Backpropagate;

}  // namespace pooling
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_POOLING_OPERATORS_H_
