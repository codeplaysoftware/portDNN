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
#ifndef PORTDNN_INCLUDE_CONV2D_CONV_TYPE_H_
#define PORTDNN_INCLUDE_CONV2D_CONV_TYPE_H_

/**
 * \file
 * Contains the declarations of the \ref sycldnn::conv2d::conv_type::Forward,
 * \ref sycldnn::conv2d::conv_type::InputBackprop and
 * \ref sycldnn::conv2d::conv_type::FilterBackprop tag types.
 *
 */
namespace sycldnn {
namespace conv2d {
/**
 * The possible types of convolution to run. Either the forward pass, the input
 * backprop or the filter backprop.
 */
namespace conv_type {

/** Tag type representing forward-only convolutions. */
struct Forward {};

/**
 * Tag type representing computation of backpropagation gradients with respect
 * to the input tensor.
 */
struct InputBackprop {};

/**
 * Tag type representing computation of backpropagation gradients with respect
 * to the filter tensor.
 */
struct FilterBackprop {};
}  // namespace conv_type
}  // namespace conv2d
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_CONV2D_CONV_TYPE_H_
