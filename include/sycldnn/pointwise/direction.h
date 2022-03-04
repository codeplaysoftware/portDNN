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

#ifndef SYCLDNN_INCLUDE_POINTWISE_DIRECTION_H_
#define SYCLDNN_INCLUDE_POINTWISE_DIRECTION_H_

/**
 * \file
 * Contains the declarations of the Forward, Gradient, and GradGrad
 * tag types.
 */

namespace sycldnn {
namespace pointwise {

struct Forward;

struct Gradient;

struct GradGrad;

}  // namespace pointwise
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_POINTWISE_DIRECTION_H_
