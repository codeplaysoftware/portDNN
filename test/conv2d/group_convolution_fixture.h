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
#ifndef PORTDNN_TEST_CONV2D_GROUPS_FIXTURE_H_
#define PORTDNN_TEST_CONV2D_GROUPS_FIXTURE_H_
#include <gtest/gtest.h>
#include <vector>

#include "portdnn/conv2d/launch.h"
#include "portdnn/conv2d/params.h"
#include "portdnn/conv2d/sizes.h"
#include "portdnn/conv2d/workspace_size.h"

#include "portdnn/transpose/launch.h"

#include "portdnn/helpers/padding.h"
#include "portdnn/helpers/scope_exit.h"

#include "test/backend/backend_test_fixture.h"
#include "test/gen/iota_initialised_data.h"
#include "test/helpers/float_comparison.h"
#include "test/helpers/transpose.h"

#include <vector>

template <typename ConvType>
struct transpose_helper {
  // Transpose input data to \p params.input_format.
  template <typename T>
  std::vector<T>& transpose_input(const sycldnn::conv2d::Conv2DParams& params,
                                  std::vector<T>& inputData,
                                  std::vector<T>& trInputData) {
    if (params.input_format == sycldnn::DataFormat::NHWC &&
        params.group_format == sycldnn::BatchFormat::INTERLEAVED) {
      transpose(trInputData, inputData,
                params.in_cols * params.in_rows * params.batch, params.groups,
                params.channels / params.groups);
      return trInputData;
    }
    return inputData;
  }

  // Generic case to transpose the filter data to params.filter_format.
  template <typename T>
  std::vector<T>& transpose_filter(const sycldnn::conv2d::Conv2DParams& params,
                                   std::vector<T>& filterData,
                                   std::vector<T>& trFilterData) {
    if (params.filter_format == sycldnn::FilterFormat::FHWC) {
      transpose(trFilterData, filterData, 1,
                params.window_cols * params.window_rows * params.channels /
                    params.groups,
                params.features);
      return trFilterData;
    } else if (params.filter_format == sycldnn::FilterFormat::HWCF &&
               params.group_format == sycldnn::BatchFormat::INTERLEAVED) {
      transpose(trFilterData, filterData,
                params.window_cols * params.window_rows * params.channels /
                    params.groups,
                params.groups, params.features / params.groups);
      return trFilterData;
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

template <typename Triple>
struct ConvolutionFixture
    : public BackendTestFixture<typename Triple::ThirdType> {
  using SelectorType = typename Triple::FirstType;
  using DataType = typename Triple::SecondType;
  using Backend = typename Triple::ThirdType;

 protected:
  /** Test a convolution with both input and filter set to `1, 2, 3,...` */
  template <typename ConvType>
  void test_conv(std::vector<DataType> const& nhwc_exp,
                 sycldnn::conv2d::Conv2DParams params,
                 DataType max_val = static_cast<DataType>(0)) {
    auto conv_sizes = sycldnn::conv2d::get_sizes<ConvType>(params);
    ASSERT_EQ(conv_sizes.output_size, nhwc_exp.size());

    transpose_helper<ConvType> helper;
    SelectorType selector{};

    std::vector<DataType> inputData =
        iota_initialised_data(conv_sizes.input_size, max_val);
    std::vector<DataType> trInputData;

    std::vector<DataType>& input =
        helper.transpose_input(params, inputData, trInputData);
    std::vector<DataType> filterData =
        iota_initialised_data(conv_sizes.filter_size, max_val);
    std::vector<DataType> trFilterData;
    std::vector<DataType>& filter =
        helper.transpose_filter(params, filterData, trFilterData);
    auto workspace_size =
        sycldnn::conv2d::query_workspace_size<ConvType>(params, selector);

    std::vector<DataType> outputData(conv_sizes.output_size,
                                     static_cast<DataType>(0));

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    auto inp_gpu = provider.get_initialised_device_memory(input.size(), input);
    auto fil_gpu =
        provider.get_initialised_device_memory(filter.size(), filter);
    auto out_gpu =
        provider.get_initialised_device_memory(outputData.size(), outputData);
    auto workspace_gpu =
        backend.template allocate<DataType>(workspace_size.recommended_size);
    SNN_ON_SCOPE_EXIT {
      backend.get_queue().wait_and_throw();
      provider.deallocate_ptr(inp_gpu);
      provider.deallocate_ptr(fil_gpu);
      provider.deallocate_ptr(out_gpu);
      provider.deallocate_ptr(workspace_gpu);
    };

    if ((params.group_format == sycldnn::BatchFormat::INTERLEAVED) &&
        !sycldnn::backend::supports_interleaved_matmul<Backend>::value) {
      // Do not run if backend does not support interleaved.
      GTEST_SKIP() << "Skipping test because backend does not support "
                      "interleaved matmul.";
    }

    if (selector.template select<ConvType>(params) ==
        sycldnn::conv2d::Algorithm::NotSupported) {
      // Do not run the test if the implementation is not supported.
      GTEST_SKIP()
          << "Skipping test because the implementation is not supported";
    }
    try {
      auto status = sycldnn::conv2d::launch<DataType, ConvType>(
          inp_gpu, fil_gpu, out_gpu, params, selector, backend, workspace_gpu,
          workspace_size.recommended_size);

      if (status.status == sycldnn::StatusCode::InvalidAlgorithm) {
        // Do not check results if the implementation is not supported.
        GTEST_SKIP()
            << "Skipping test because the selected convolution algorithm "
               "does not support group convolution.";
      }
      ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
      status.event.wait_and_throw();
    } catch (cl::sycl::exception const& e) {
      std::cerr << "Caught SYCL exception:\n" << e.what() << std::endl;
      throw;
    }

    provider.copy_device_data_to_host(outputData.size(), out_gpu, outputData);

    for (size_t i = 0; i < nhwc_exp.size(); ++i) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      SNN_ALMOST_EQUAL(nhwc_exp[i], outputData[i], 10u);
    }
  }
};

template <typename Tuple, int Window, int Stride, int Groups>
struct GroupWindowStrideTest : public ConvolutionFixture<Tuple> {
  using typename ConvolutionFixture<Tuple>::DataType;
  using Conv2DParams = sycldnn::conv2d::Conv2DParams;

 protected:
  void run_forward_test(std::vector<DataType> const& exp_out,
                        std::array<int, 4> const& in_shape, int features,
                        sycldnn::PaddingMode padding,
                        sycldnn::FilterFormat filter_format,
                        sycldnn::BatchFormat group_format, DataType max_val) {
    auto params =
        get_params(in_shape, features, padding, filter_format, group_format);
    test_forward(exp_out, params, max_val);
  }

 private:
  Conv2DParams get_params(std::array<int, 4> const& in_shape, int features,
                          sycldnn::PaddingMode padding,
                          sycldnn::FilterFormat filter_format,
                          sycldnn::BatchFormat group_format) {
    sycldnn::conv2d::Conv2DParams params{};
    params.channels = in_shape[3];
    params.features = features;
    params.batch = in_shape[0];
    params.in_rows = in_shape[1];
    params.in_cols = in_shape[2];
    params.window_rows = Window;
    params.window_cols = Window;
    params.stride_rows = Stride;
    params.stride_cols = Stride;
    params.dilation_rows = 1;
    params.dilation_cols = 1;
    params.filter_format = filter_format;
    params.input_format = sycldnn::DataFormat::NHWC;
    params.group_format = group_format;
    params.groups = Groups;
    return sycldnn::helpers::add_padding_to(params, padding);
  }
  void test_forward(std::vector<DataType> const& exp,
                    Conv2DParams const& params, DataType max_val) {
    using sycldnn::conv2d::conv_type::Forward;
    SCOPED_TRACE("Forward pass");
    this->template test_conv<Forward>(exp, params, max_val);
  }
};
#endif  // PORTDNN_TEST_CONV2D_GROUPS_FIXTURE_H_
