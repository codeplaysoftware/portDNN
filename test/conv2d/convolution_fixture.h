/*
 * Copyright 2018 Codeplay Software Ltd.
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
#ifndef SYCLDNN_TEST_CONV2D_CONVOLUTION_FIXTURE_H_
#define SYCLDNN_TEST_CONV2D_CONVOLUTION_FIXTURE_H_

#include <gtest/gtest.h>
#include <vector>

#include "sycldnn/conv2d/launch.h"
#include "sycldnn/conv2d/params.h"
#include "sycldnn/conv2d/sizes.h"
#include "test/gen/generated_test_fixture.h"
#include "test/gen/iota_initialised_data.h"

template <typename Triple>
struct ConvolutionFixture
    : public GeneratedTestFixture<typename Triple::SecondType,
                                  typename Triple::ThirdType> {
  using SelectorType = typename Triple::FirstType;
  using DataType = typename Triple::SecondType;
  using Backend = typename Triple::ThirdType;

 protected:
  /** Test a convolution with both input and filter set to `1, 2, 3,...` */
  template <typename ConvType>
  void test_conv(std::vector<DataType> exp,
                 sycldnn::conv2d::Conv2DParams const& params,
                 DataType max_val = static_cast<DataType>(0)) {
    auto conv_sizes = sycldnn::conv2d::get_sizes<ConvType>(params);
    ASSERT_EQ(conv_sizes.output_size, exp.size());

    std::vector<DataType> input =
        iota_initialised_data(conv_sizes.input_size, max_val);
    std::vector<DataType> filter =
        iota_initialised_data(conv_sizes.filter_size, max_val);
    std::vector<DataType> output(conv_sizes.output_size,
                                 static_cast<DataType>(0));

    auto inp_gpu =
        this->get_initialised_device_memory(conv_sizes.input_size, input);
    auto fil_gpu =
        this->get_initialised_device_memory(conv_sizes.filter_size, filter);
    auto out_gpu =
        this->get_initialised_device_memory(conv_sizes.output_size, output);

    SelectorType selector{};
    if (selector.select(params) == sycldnn::conv2d::Algorithm::NotSupported) {
      // Do not run the test if the implementation is not supported.
      return;
    }
    try {
      auto status = sycldnn::conv2d::launch<DataType, ConvType>(
          inp_gpu, fil_gpu, out_gpu, params, selector, this->backend_);

      if (status.status == sycldnn::StatusCode::InvalidAlgorithm) {
        // Do not check results if the implementation is not supported.
        return;
      }
      ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
      status.event.wait_and_throw();
    } catch (cl::sycl::exception const& e) {
      std::cerr << "Caught SYCL exception:\n" << e.what() << std::endl;
      throw;
    }
    this->copy_device_data_to_host(conv_sizes.output_size, out_gpu, output);
    this->deallocate_ptr(inp_gpu);
    this->deallocate_ptr(fil_gpu);
    this->deallocate_ptr(out_gpu);

    for (size_t i = 0; i < exp.size(); ++i) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      if (std::is_same<DataType, double>::value) {
        EXPECT_DOUBLE_EQ(exp[i], output[i]);
      } else {
        EXPECT_FLOAT_EQ(exp[i], output[i]);
      }
    }
  }
};

#endif  // SYCLDNN_TEST_CONV2D_CONVOLUTION_FIXTURE_H_
