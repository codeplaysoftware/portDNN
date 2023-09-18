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

#ifndef PORTDNN_INCLUDE_POINTWISE_OPERATORS_H_
#define PORTDNN_INCLUDE_POINTWISE_OPERATORS_H_
/**
 * \file
 * Contains the declarations of the \ref sycldnn::pointwise::Relu
 * and \ref sycldnn::pointwise::Tanh and \ref sycldnn::pointwise::Exp
 * tag types.
 */

namespace sycldnn {
namespace pointwise {

template <typename Direction>
struct Relu;

template <typename Direction>
struct Tanh;

template <typename Direction>
struct Exp;

template <typename Direction>
struct Log;

template <typename Direction>
struct Floor;

template <typename Direction>
struct Sqrt;

}  // namespace pointwise
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_POINTWISE_OPERATORS_H_
