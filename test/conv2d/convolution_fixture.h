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
#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>
#include <vector>
#include "sycldnn/conv2d/launch.h"
#include "sycldnn/conv2d/params.h"
#include "sycldnn/conv2d/sizes.h"
#include "test/backend/eigen_backend_test_fixture.h"

template <typename Pair>
struct ConvolutionFixture : public EigenBackendTest {
  using SelectorType = typename Pair::FirstType;
  using DataType = typename Pair::SecondType;

 protected:
  /** Test a convolution with both input and filter set to `1, 2, 3,...` */
  template <typename ConvType>
  void test_conv(std::vector<DataType> exp,
                 sycldnn::conv2d::Conv2DParams const& params,
                 DataType max_val = static_cast<DataType>(0)) {
    auto conv_sizes = sycldnn::conv2d::get_sizes<ConvType>(params);
    ASSERT_EQ(conv_sizes.output_size, exp.size());

    std::vector<DataType> input;
    iota_n_modulo(input, conv_sizes.input_size, static_cast<DataType>(1),
                  max_val);
    ASSERT_EQ(conv_sizes.input_size, input.size());

    std::vector<DataType> filter;
    iota_n_modulo(filter, conv_sizes.filter_size, static_cast<DataType>(1),
                  max_val);
    ASSERT_EQ(conv_sizes.filter_size, filter.size());

    size_t inp_bytes = conv_sizes.input_size * sizeof(exp[0]);
    DataType* inp_gpu = static_cast<DataType*>(device_.allocate(inp_bytes));
    device_.memcpyHostToDevice(inp_gpu, input.data(), inp_bytes);

    size_t fil_bytes = conv_sizes.filter_size * sizeof(exp[0]);
    DataType* fil_gpu = static_cast<DataType*>(device_.allocate(fil_bytes));
    device_.memcpyHostToDevice(fil_gpu, filter.data(), fil_bytes);

    size_t out_bytes = conv_sizes.output_size * sizeof(exp[0]);
    DataType* out_gpu = static_cast<DataType*>(device_.allocate(out_bytes));

    SelectorType selector{};
    if (selector.select(params) == sycldnn::conv2d::Algorithm::NotSupported) {
      // Do not run the test if the implementation is not supported.
      return;
    }
    auto status = sycldnn::conv2d::launch<DataType, ConvType>(
        inp_gpu, fil_gpu, out_gpu, params, selector, backend_);

    if (status.status == sycldnn::StatusCode::InvalidAlgorithm) {
      // Do not check results if the implementation is not supported.
      return;
    }
    ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
    status.event.wait();

    std::vector<DataType> output;
    output.resize(conv_sizes.output_size);
    device_.memcpyDeviceToHost(output.data(), out_gpu, out_bytes);

    for (size_t i = 0; i < exp.size(); ++i) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      if (std::is_same<DataType, double>::value) {
        EXPECT_DOUBLE_EQ(exp[i], output[i]);
      } else {
        EXPECT_FLOAT_EQ(exp[i], output[i]);
      }
    }
    device_.deallocate_all();
  }

 private:
  /**
   * Fill a vector with the values:
   *   `init_value, init_value+1, ..., max_value-1, max_value, init_value,...`
   * where the values will increase by `1` each step, but the values are
   * limited by `max_value`. Once `max_value` is reached, the values begin
   * again at init_value.
   */
  template <typename T>
  void iota_n_modulo(std::vector<T>& c, size_t size, T init_value,
                     T max_value) {
    if (max_value < 1) {
      return iota_n(c, size, init_value);
    }
    c.reserve(size);
    // Want the max value to ba attained, so need to add an additional step.
    size_t n_steps = static_cast<size_t>(max_value - init_value) + 1;
    size_t n_done = 0;
    while (n_done < size) {
      size_t to_do = size - n_done;
      size_t this_time = to_do > n_steps ? n_steps : to_do;
      iota_n(c, this_time, init_value);
      n_done += this_time;
    }
  }
  /** Fill a vector with `value, value+1,...` with `size` elements. */
  template <typename T>
  void iota_n(std::vector<T>& c, size_t size, T value) {
    c.reserve(size);
    std::generate_n(std::back_inserter(c), size, [&value] { return value++; });
  }
};

#endif  // SYCLDNN_TEST_CONV2D_CONVOLUTION_FIXTURE_H_
