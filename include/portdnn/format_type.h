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
#ifndef PORTDNN_INCLUDE_FORMAT_TYPE_H_
#define PORTDNN_INCLUDE_FORMAT_TYPE_H_

#include "portdnn/data_format.h"
#include "portdnn/filter_format.h"

/**
 * \file
 * Declare \ref sycldnn::DataFormat and \ref sycldnn::FilterFormat as types that
 * can be used to specialise SYCL kernels. DataFormat and FilterFormat are tied
 * to the same type to avoid combinatorial explosion.
 */
namespace sycldnn {
namespace layout {

/**
 * \brief Tie NHWC input format and HWCF filter format.
 */
struct NHWC {
  /**
   * \brief Layout to use for the input of most operations.
   */
  static constexpr DataFormat input_layout = DataFormat::NHWC;

  /**
   * \brief Layout to use for the filter input of conv2d.
   */
  static constexpr FilterFormat filter_layout = FilterFormat::HWCF;
};

/**
 * \brief Tie NCHW input format and FCHW filter format.
 */
struct NCHW {
  /**
   * \brief Layout to use for the input of most operations.
   */
  static constexpr DataFormat input_layout = DataFormat::NCHW;

  /**
   * \brief Layout to use for the filter input of conv2d.
   */
  static constexpr FilterFormat filter_layout = FilterFormat::FCHW;
};

}  // namespace layout
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_FORMAT_TYPE_H_
