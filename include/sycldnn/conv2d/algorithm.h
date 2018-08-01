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
#ifndef SYCLDNN_INCLUDE_CONV2D_ALGORITHM_ALGORITHM_H_
#define SYCLDNN_INCLUDE_CONV2D_ALGORITHM_ALGORITHM_H_

/**
 * \file
 * Contains the declaration of the \ref sycldnn::conv2d::Algorithm enumerated
 * type.
 * This type is used to describe the various supported convolution algorithms.
 */
namespace sycldnn {
namespace conv2d {
/**
 * The implemented 2d convolution algorithms.
 */
enum class Algorithm {
  /** Fallback not supported algorithm tag. */
  NotSupported,
  /** Direct convolution using a naive implementation. */
  Direct,
  /** Tiled approach to maximise data reuse within a thread. */
  Tiled,
  /** Im2col implementation with temporary buffer. */
  Im2col,
  /** Winograd implementation with temporary buffer. */
  Winograd,
};
}  // namespace conv2d
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_CONV2D_ALGORITHM_ALGORITHM_H_
