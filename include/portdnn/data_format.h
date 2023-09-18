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
#ifndef PORTDNN_INCLUDE_DATA_FORMAT_H_
#define PORTDNN_INCLUDE_DATA_FORMAT_H_

/**
 * \file
 * Contains the declaration of the \ref sycldnn::DataFormat enumerated
 * type. This type is used to specify the memory layout used for a given
 * tensor.
 */
namespace sycldnn {

/**
 * Suppose that for a given tensor, you have batches (N), image width (W),
 * image height (H), and channels (C). Then the formats available are:
 */
enum class DataFormat {
  /**
   * DataFormat::NHWC where batches are the outer-most dimension, followed
   * by image height, then image width, then channels.
   */
  NHWC,

  /**
   * DataFormat::NCHW where batches are the outer-most dimension, followed
   * by channels, then image height, then image width.
   */
  NCHW
};
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_DATA_FORMAT_H_
