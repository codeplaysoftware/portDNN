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

#ifndef SYCLDNN_INCLUDE_POINTWISE_OPERATORS_H_
#define SYCLDNN_INCLUDE_POINTWISE_OPERATORS_H_
/**
 * \file
 * Contains the declarations of the \ref sycldnn::pointwise::Relu
 * and \ref sycldnn::pointwise::Tanh, sycldnn::pointwise::Exp,
 * sycldnn::pointwise::SoftMaxDiv and \ref sycldnn::pointwise::Batchnorm_MeanDiv
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
struct SoftMaxDiv;

template <typename Direction>
struct Batchnorm_MeanDiv;

}  // namespace pointwise
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_POINTWISE_OPERATORS_H_
