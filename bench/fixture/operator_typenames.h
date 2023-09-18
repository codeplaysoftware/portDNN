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
#ifndef PORTDNN_BENCH_FIXTURE_OPERATOR_TYPENAMES_H_
#define PORTDNN_BENCH_FIXTURE_OPERATOR_TYPENAMES_H_

#include "portdnn/pointwise/operators.h"
#include "portdnn/pooling/operators.h"

namespace sycldnn {
namespace bench {

/** Provides a typename for a templated operator class. */
template <template <typename> class Operator>
struct OperatorTypeName {
  static const char* const name;
};

// Templated Pooling operators
template <>
constexpr const char* OperatorTypeName<sycldnn::pooling::Max>::name = "Max";

template <>
constexpr const char* OperatorTypeName<sycldnn::pooling::Average>::name =
    "Average";

// Templated Pointwise operators
template <>
constexpr const char* OperatorTypeName<sycldnn::pointwise::Relu>::name = "Relu";

template <>
constexpr const char* OperatorTypeName<sycldnn::pointwise::Tanh>::name = "Tanh";

}  // namespace bench
}  // namespace sycldnn

#endif  // PORTDNN_BENCH_FIXTURE_OPERATOR_TYPENAMES_H_
