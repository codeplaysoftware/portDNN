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

#ifndef PORTDNN_INCLUDE_BINARYOP_OPERATORS_H_
#define PORTDNN_INCLUDE_BINARYOP_OPERATORS_H_
/**
 * \file
 * Contains the declarations of the sycldnn::binaryop::Add,
 * sycldnn::binaryop::Sub, sycldnn::binaryop::Mul and sycldnn::binaryop::Div tag
 * types.
 */

namespace sycldnn {
namespace binaryop {

struct Add;

struct Sub;

struct Mul;

struct Div;

}  // namespace binaryop
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_BINARYOP_OPERATORS_H_
