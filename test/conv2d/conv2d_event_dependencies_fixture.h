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
#ifndef PORTDNN_TEST_CONV2D_CONV2D_DEPEND_FIXTURE_H_
#define PORTDNN_TEST_CONV2D_CONV2D_DEPEND_FIXTURE_H_

#include <gtest/gtest.h>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <CL/sycl.hpp>

#include "portdnn/backend/snn_usm_backend.h"

#include "portdnn/helpers/scope_exit.h"

#include "portdnn/conv2d/launch.h"
#include "portdnn/conv2d/workspace_size.h"

#include "test/backend/backend_test_fixture.h"
#include "test/gen/iota_initialised_data.h"
#include "test/helpers/conv2d_transpose_helper.h"
#include "test/helpers/dependency_check.h"

template <typename Tuple>
struct Conv2DEventFixture : public BackendTestFixture<typename Tuple::T2> {
  using SelectorType = typename Tuple::T0;
  using DataType = typename Tuple::T1;
  using Backend = typename Tuple::T2;
  static constexpr sycldnn::DataFormat input_format = Tuple::T3::input_layout;
  static constexpr sycldnn::FilterFormat filter_format =
      Tuple::T3::filter_layout;

 protected:
  template <typename ConvType>
  void run(sycldnn::conv2d::Conv2DParams params,
           DataType max_val = static_cast<DataType>(0),
           size_t input_offset = 0u, size_t filter_offset = 0u,
           size_t output_offset = 0u) {
    ASSERT_EQ(params.input_format, sycldnn::DataFormat::NHWC)
        << "Tests should be written for NHWC convolutions. The input layout is "
           "set from the fixture type.";
    ASSERT_EQ(params.filter_format, sycldnn::FilterFormat::HWCF)
        << "Tests should be written for HWCF convolutions. The filter layout "
           "is set from the fixture type.";
    ASSERT_TRUE(input_format == sycldnn::DataFormat::NHWC ||
                input_format == sycldnn::DataFormat::NCHW);
    ASSERT_TRUE(filter_format == sycldnn::FilterFormat::HWCF ||
                filter_format == sycldnn::FilterFormat::FCHW);

    SelectorType selector{};

    params.input_format = input_format;
    params.filter_format = filter_format;
    auto conv_batch_sizes = sycldnn::conv2d::get_batch_sizes<ConvType>(params);
    auto conv_spatial_sizes =
        sycldnn::conv2d::get_spatial_sizes<ConvType>(params);
    auto conv_channel_sizes =
        sycldnn::conv2d::get_channel_sizes<ConvType>(params);
    auto conv_sizes = sycldnn::conv2d::get_sizes<ConvType>(params);
    transpose_helper<ConvType> helper;

    auto workspace_size =
        sycldnn::conv2d::query_workspace_size<ConvType>(params, selector);

    std::vector<DataType> inputData =
        iota_initialised_data(conv_sizes.input_size, max_val);
    std::vector<DataType> trInputData;
    std::vector<DataType>& input =
        helper.transpose_input(params, inputData, trInputData, conv_batch_sizes,
                               conv_spatial_sizes, conv_channel_sizes);
    input.insert(input.begin(), input_offset, 0);

    std::vector<DataType> filterData =
        iota_initialised_data(conv_sizes.filter_size, max_val);
    std::vector<DataType> trFilterData;
    std::vector<DataType>& filter = helper.transpose_filter(
        params, filterData, trFilterData, conv_batch_sizes, conv_spatial_sizes,
        conv_channel_sizes);
    filter.insert(filter.begin(), filter_offset, 0);

    std::vector<DataType> outputData(conv_sizes.output_size + output_offset,
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
      provider.deallocate_ptr(inp_gpu);
      provider.deallocate_ptr(fil_gpu);
      provider.deallocate_ptr(out_gpu);
      provider.deallocate_ptr(workspace_gpu);
    };

    if (selector.template select<ConvType>(params) ==
        sycldnn::conv2d::Algorithm::NotSupported) {
      // Do not run the test if the implementation is not supported.
      GTEST_SKIP() << "Skipping test because implementation is not supported.";
    }
    try {
      dependency_test_params dep_test_params;
      cl::sycl::event dependee_e = create_event(backend, dep_test_params);
      auto status = sycldnn::conv2d::launch<DataType, ConvType>(
          inp_gpu + input_offset, fil_gpu + filter_offset,
          out_gpu + output_offset, params, selector, backend, workspace_gpu,
          workspace_size.recommended_size,
          std::vector<cl::sycl::event>{dependee_e});

      if (status.status == sycldnn::StatusCode::InvalidAlgorithm) {
        // Do not check results if the implementation is not supported.
        GTEST_SKIP()
            << "Skipping test because the selected convolution algorithm "
               "does not support the provided parameters.";
      }
      check_dependency(dependee_e, status.event, backend, dep_test_params);

      status.event.wait_and_throw();
    } catch (cl::sycl::exception const& e) {
      std::cerr << "Caught SYCL exception:\n" << e.what() << std::endl;
      throw;
    }
  }
};

#endif  // PORTDNN_TEST_CONV2D_CONV2D_DEPEND_FIXTURE_H_
