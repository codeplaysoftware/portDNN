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
#ifndef PORTDNN_INCLUDE_FILTER_FORMAT_H_
#define PORTDNN_INCLUDE_FILTER_FORMAT_H_

/**
 * \file
 * Contains the declaration of the \ref sycldnn::FilterFormat enumerated
 * type. This type is used to specify the memory layout used for a given
 * filter tensor.
 */
namespace sycldnn {

/**
 * Suppose that for a given tensor, you have input feature maps, or channels
 * (C), output feature maps, or features (F), filter height (H), and filter
 * width (W). Then the formats available are:
 */
enum class FilterFormat {
  /**
   * FilterFormat::HWCF where the filter height is the outer-most dimension,
   * followed by filter width, then input feature maps, then output
   * feature maps.
   */
  HWCF,

  /**
   * FilterFormat::FCHW where the output feature maps are the outer-most
   * dimension, followed by input feature maps, then filter height, then
   * filter width.
   */
  FCHW,

  /**
   * FilterFormat::FHWC where the output feature maps are the outer-most
   * dimension, followed by filter height, then filter width, then input
   * feature maps.
   */
  FHWC
};
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_FILTER_FORMAT_H_
