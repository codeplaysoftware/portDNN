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
#ifndef SYCLDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_TRANSPOSE_HELPER_H_
#define SYCLDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_TRANSPOSE_HELPER_H_

#include "sycldnn/conv2d/conv_type.h"
#include "sycldnn/conv2d/params.h"

#include "sycldnn/internal/conv2d/im2col/transform_sizes.h"
#include "sycldnn/internal/transpose/launch.h"

namespace sycldnn {
namespace conv2d {
namespace internal {
namespace im2col {

template <typename ConvType>
struct BaseIm2colTransposeHelper {
  static bool should_transpose_input(Conv2DParams const& params) {
    bool should_trans = params.input_format == sycldnn::DataFormat::NCHW;
    should_trans =
        should_trans && params.window_rows == 1 && params.window_cols == 1;
    should_trans =
        should_trans && params.stride_rows == 1 && params.stride_cols == 1;
    should_trans =
        should_trans && params.dilation_rows == 1 && params.dilation_cols == 1;
    should_trans = should_trans && params.pad_rows == 0 && params.pad_cols == 0;
    return should_trans;
  }

  template <typename T, template <typename> class MemObj>
  static SNNStatus transpose_output(
      MemObj<const T>& in_mem_obj, MemObj<T>& out_mem_obj, int batch,
      int groups, int out_channel, int out_rows, int out_cols,
      cl::sycl::queue queue, std::vector<cl::sycl::event> const& events) {
    if (groups == 1) {
      std::vector<int> const FNHW_TO_NFHW = {1, 0, 2, 3};
      return sycldnn::transpose::internal::launch(
          in_mem_obj, out_mem_obj, {out_channel, batch, out_rows, out_cols},
          FNHW_TO_NFHW, queue, events);
    } else {
      const std::vector<int> GNHWC_TO_NHWGC = {1, 2, 0, 3};
      return sycldnn::transpose::internal::launch(
          in_mem_obj, out_mem_obj,
          {groups, batch, out_rows * out_cols, out_channel}, GNHWC_TO_NHWGC,
          queue, events);
    }
  }
};

template <typename T, typename ConvType>
struct Im2colTransposeHelper : BaseIm2colTransposeHelper<ConvType> {};

template <typename T>
struct Im2colTransposeHelper<T, conv2d::conv_type::Forward>
    : BaseIm2colTransposeHelper<conv2d::conv_type::Forward> {
  using BaseIm2colTransposeHelper<
      conv2d::conv_type::Forward>::should_transpose_input;

  template <template <typename> class MemObj>
  static SNNStatus transpose_input(MemObj<const T>& in_mem_obj,
                                   MemObj<T>& out_mem_obj,
                                   Conv2DParams const& params,
                                   cl::sycl::queue queue,
                                   std::vector<cl::sycl::event> const& events) {
    std::vector<int> const NCHW_TO_CNHW = {1, 0, 2};
    return sycldnn::transpose::internal::launch(
        in_mem_obj, out_mem_obj,
        {params.batch, params.channels, params.in_cols * params.in_rows},
        NCHW_TO_CNHW, queue, events);
  }

  template <template <typename> class MemObj>
  static SNNStatus transpose_filter(
      MemObj<const T>& fil_mem_obj, MemObj<T>& out_mem_obj,
      Conv2DParams const& params, cl::sycl::queue queue,
      std::vector<cl::sycl::event> const& events) {
    SNN_VALIDATE_PARAM(
        params.group_format != sycldnn::BatchFormat::INTERLEAVED ||
            params.filter_format == sycldnn::FilterFormat::HWCF,
        "Interleaved group format is only supported for HWCF filter format.");

    int const features_per_group = params.features / params.groups;
    int const channels_per_group = params.channels / params.groups;
    std::vector<int> const HWCGF_TO_HWCFG = {3, 0, 1, 2, 4};
    return sycldnn::transpose::internal::launch(
        fil_mem_obj, out_mem_obj,
        {params.window_rows, params.window_cols, channels_per_group,
         params.groups, features_per_group},
        HWCGF_TO_HWCFG, queue, events);
  }

  template <template <typename> class MemObj>
  static SNNStatus transpose_output(
      MemObj<const T>& in_mem_obj, MemObj<T>& out_mem_obj,
      Conv2DParams const& params, cl::sycl::queue queue,
      std::vector<cl::sycl::event> const& events) {
    auto const out_channels = params.features / params.groups;
    auto const out_rows = params.out_rows;
    auto const out_cols = params.out_cols;
    // FHWN -> NFHW
    return BaseIm2colTransposeHelper<conv2d::conv_type::Forward>::
        template transpose_output<T>(in_mem_obj, out_mem_obj, params.batch,
                                     params.groups, out_channels, out_rows,
                                     out_cols, queue, events);
  }
};

template <typename T>
struct Im2colTransposeHelper<T, conv2d::conv_type::InputBackprop>
    : BaseIm2colTransposeHelper<conv2d::conv_type::InputBackprop> {
  using BaseIm2colTransposeHelper<
      conv2d::conv_type::InputBackprop>::should_transpose_input;

  template <template <typename> class MemObj>
  static SNNStatus transpose_input(MemObj<const T>& in_mem_obj,
                                   MemObj<T>& out_mem_obj,
                                   Conv2DParams const& params,
                                   cl::sycl::queue queue,
                                   std::vector<cl::sycl::event> const& events) {
    std::vector<int> const NFHW_TO_FHWN = {1, 0, 2};
    return sycldnn::transpose::internal::launch(
        in_mem_obj, out_mem_obj,
        {params.batch, params.features, params.in_cols * params.in_rows},
        NFHW_TO_FHWN, queue, events);
  }

  template <template <typename> class MemObj>
  static SNNStatus transpose_output(
      MemObj<const T>& in_mem_obj, MemObj<T>& out_mem_obj,
      Conv2DParams const& params, cl::sycl::queue queue,
      std::vector<cl::sycl::event> const& events) {
    auto const out_channels = params.channels / params.groups;
    auto const out_rows = params.in_rows;
    auto const out_cols = params.in_cols;
    // CHWN -> NCHW
    return BaseIm2colTransposeHelper<conv2d::conv_type::InputBackprop>::
        template transpose_output<T>(in_mem_obj, out_mem_obj, params.batch,
                                     params.groups, out_channels, out_rows,
                                     out_cols, queue, events);
  }
};

template <typename T>
struct Im2colTransposeHelper<T, conv2d::conv_type::FilterBackprop>
    : BaseIm2colTransposeHelper<conv2d::conv_type::FilterBackprop> {
  static bool should_transpose_input(Conv2DParams const& params) {
    bool should_trans = params.input_format == sycldnn::DataFormat::NCHW;
    should_trans = should_trans && params.out_rows == 1 && params.out_cols == 1;
    should_trans =
        should_trans && params.stride_rows == 1 && params.stride_cols == 1;
    should_trans =
        should_trans && params.dilation_rows == 1 && params.dilation_cols == 1;
    should_trans = should_trans && params.pad_rows == 0 && params.pad_cols == 0;
    return should_trans;
  }

  template <template <typename> class MemObj>
  static SNNStatus transpose_input(MemObj<const T>& in_mem_obj,
                                   MemObj<T>& out_mem_obj,
                                   Conv2DParams const& params,
                                   cl::sycl::queue queue,
                                   std::vector<cl::sycl::event> const& events) {
    std::vector<int> const NCHW_TO_NHWC = {0, 2, 1};
    return sycldnn::transpose::internal::launch(
        in_mem_obj, out_mem_obj,
        {params.batch, params.channels, params.in_cols * params.in_rows},
        NCHW_TO_NHWC, queue, events);
  }

  template <template <typename> class MemObj>
  static SNNStatus transpose_filter(
      MemObj<const T>& fil_mem_obj, MemObj<T>& out_mem_obj,
      Conv2DParams const& params, cl::sycl::queue queue,
      std::vector<cl::sycl::event> const& events) {
    // When NCHW and filter backprop is computed we need to do and additional
    // transpose to the filter in order to re-organize the data for im2col.
    std::vector<int> const NFHW_TO_FNHW = {1, 0, 2};
    return sycldnn::transpose::internal::launch(
        fil_mem_obj, out_mem_obj,
        {params.batch, params.features, params.out_rows * params.out_cols},
        NFHW_TO_FNHW, queue, events);
  }
};
}  // namespace im2col
}  // namespace internal
}  // namespace conv2d
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_INTERNAL_CONV2D_IM2COL_TRANSPOSE_HELPER_H_