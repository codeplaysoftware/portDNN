/*
 * Copyright Codeplay Software Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use these files except in compliance with the License.
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
#ifndef PORTDNN_BENCH_FIXTURE_TYPENAMES_H_
#define PORTDNN_BENCH_FIXTURE_TYPENAMES_H_

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/pointwise/direction.h"
#include "portdnn/pooling/operators.h"

namespace sycldnn {
namespace bench {

/** Provides a typename for a concrete class or type. */
template <typename>
struct TypeName {
  static const char* const name;
};

// Default typename in case of unrecognised class
template <typename T>
constexpr const char* TypeName<T>::name = "Unknown";

// Direction for Pooling
template <>
constexpr const char* TypeName<sycldnn::pooling::Forward>::name = "Forward";

template <>
constexpr const char* TypeName<sycldnn::pooling::Backpropagate>::name =
    "Backpropagate";

// Direction for Pointwise
template <>
constexpr const char* TypeName<sycldnn::pointwise::Forward>::name = "Forward";

template <>
constexpr const char* TypeName<sycldnn::pointwise::Gradient>::name = "Gradient";

template <>
constexpr const char* TypeName<sycldnn::pointwise::GradGrad>::name = "GradGrad";

// Types of convolution
template <>
constexpr const char* TypeName<sycldnn::conv2d::conv_type::Forward>::name =
    "Forward";

template <>
constexpr const char*
    TypeName<sycldnn::conv2d::conv_type::FilterBackprop>::name =
        "FilterBackprop";

template <>
constexpr const char*
    TypeName<sycldnn::conv2d::conv_type::InputBackprop>::name = "InputBackprop";

}  // namespace bench
}  // namespace sycldnn

#endif  // PORTDNN_BENCH_FIXTURE_TYPENAMES_H_
