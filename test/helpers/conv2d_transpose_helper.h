/*
 * Copyright Codeplay Software Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use these files except in compliance with the License.
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

#ifndef PORTDNN_TEST_HELPERS_CONV2D_TRANSPOSE_H_
#define PORTDNN_TEST_HELPERS_CONV2D_TRANSPOSE_H_

#include "test/helpers/transpose.h"

#include <vector>
template <typename ConvType>
struct transpose_helper {
  /**
   * \brief Transpose input data to \p params.input_format.
   * This is shared for all ConvType as the ConvSizes are already adjusted for a
   * given ConvType.
   *
   * \param params
   * \param inputData Initialised data.
   * \param trInputData Storage to use if the data needs to be transposed.
   * \param conv_batch_sizes
   * \param conv_spatial_sizes
   * \param conv_channel_sizes
   * \return std::vector<T>* Pointer to data to use.
   */
  template <typename T>
  std::vector<T>& transpose_input(
      const sycldnn::conv2d::Conv2DParams& params, std::vector<T>& inputData,
      std::vector<T>& trInputData,
      const sycldnn::conv2d::ConvSizes& conv_batch_sizes,
      const sycldnn::conv2d::ConvSizes& conv_spatial_sizes,
      const sycldnn::conv2d::ConvSizes& conv_channel_sizes) {
    if (params.input_format == sycldnn::DataFormat::NCHW) {
      transpose(trInputData, inputData, conv_batch_sizes.input_size,
                conv_spatial_sizes.input_size, conv_channel_sizes.input_size);
      return trInputData;
    }
    return inputData;
  }

  /**
   * \brief Generic case to transpose the filter data to \p
   * params.filter_format.
   *
   * \param params
   * \param filterData Initialised data.
   * \param trFilterData Storage to use if the data needs to be transposed.
   * \param conv_batch_sizes
   * \param conv_spatial_sizes
   * \param conv_channel_sizes
   * \param filter_offset Optional offset that is not transposed.
   * \return std::vector<T>* Pointer to data to use.
   */
  template <typename T>
  std::vector<T>& transpose_filter(
      const sycldnn::conv2d::Conv2DParams& params, std::vector<T>& filterData,
      std::vector<T>& trFilterData,
      const sycldnn::conv2d::ConvSizes& conv_batch_sizes,
      const sycldnn::conv2d::ConvSizes& conv_spatial_sizes,
      const sycldnn::conv2d::ConvSizes& conv_channel_sizes,
      int filter_offset = 0) {
    if (params.filter_format == sycldnn::FilterFormat::FCHW) {
      // HWCF -> HWFC
      transpose(trFilterData, filterData, conv_spatial_sizes.filter_size,
                params.channels, params.features, filter_offset);
      // HWFC -> FCHW
      transpose(filterData, trFilterData, conv_batch_sizes.filter_size,
                conv_spatial_sizes.filter_size, conv_channel_sizes.filter_size,
                filter_offset);
    }
    return filterData;
  }

  /**
   * \brief Generic case to transpose the output data to \p params.input_format.
   *
   * \param params
   * \param outputData Initialised data.
   * \param trOutputData Storage to use if the data needs to be transposed.
   * \param conv_batch_sizes
   * \param conv_spatial_sizes
   * \param conv_channel_sizes
   * \param output_offset Optional offset that is not transposed.
   * \return std::vector<T>* Pointer to data to use.
   */
  template <typename T>
  std::vector<T>& transpose_output(
      const sycldnn::conv2d::Conv2DParams& params, std::vector<T>& outputData,
      std::vector<T>& trOutputData,
      const sycldnn::conv2d::ConvSizes& conv_batch_sizes,
      const sycldnn::conv2d::ConvSizes& conv_spatial_sizes,
      const sycldnn::conv2d::ConvSizes& conv_channel_sizes,
      int output_offset = 0) {
    if (params.input_format == sycldnn::DataFormat::NCHW) {
      transpose(trOutputData, outputData, conv_batch_sizes.output_size,
                conv_channel_sizes.output_size, conv_spatial_sizes.output_size,
                output_offset);
      return trOutputData;
    }
    return outputData;
  }
};

/**
 * \brief Filter and output transposes are swapped for FilterBackprop.
 *
 * \copydetail transpose_helper::transpose_filter
 */
template <>
template <typename T>
std::vector<T>&
transpose_helper<sycldnn::conv2d::conv_type::FilterBackprop>::transpose_filter(
    const sycldnn::conv2d::Conv2DParams& params, std::vector<T>& filterData,
    std::vector<T>& trFilterData,
    const sycldnn::conv2d::ConvSizes& conv_batch_sizes,
    const sycldnn::conv2d::ConvSizes& conv_spatial_sizes,
    const sycldnn::conv2d::ConvSizes& conv_channel_sizes, int filter_offset) {
  if (params.input_format == sycldnn::DataFormat::NCHW) {
    transpose(trFilterData, filterData, conv_batch_sizes.filter_size,
              conv_spatial_sizes.filter_size, conv_channel_sizes.filter_size,
              filter_offset);
    return trFilterData;
  }
  return filterData;
}

/**
 * \brief Filter and output transposes are swapped for FilterBackprop.
 *
 * \copydetail transpose_helper::transpose_output
 */
template <>
template <typename T>
std::vector<T>&
transpose_helper<sycldnn::conv2d::conv_type::FilterBackprop>::transpose_output(
    const sycldnn::conv2d::Conv2DParams& params, std::vector<T>& outputData,
    std::vector<T>& trOutputData,
    const sycldnn::conv2d::ConvSizes& conv_batch_sizes,
    const sycldnn::conv2d::ConvSizes& conv_spatial_sizes,
    const sycldnn::conv2d::ConvSizes& conv_channel_sizes, int output_offset) {
  if (params.filter_format == sycldnn::FilterFormat::FCHW) {
    // FCHW -> HWFC
    transpose(trOutputData, outputData, conv_batch_sizes.output_size,
              conv_channel_sizes.output_size, conv_spatial_sizes.output_size,
              output_offset);
    // HWFC -> HWCF
    transpose(outputData, trOutputData, conv_spatial_sizes.output_size,
              params.features, params.channels, output_offset);
  }
  return outputData;
}

#endif  // PORTDNN_TEST_HELPERS_CONV2D_TRANSPOSE_H_
