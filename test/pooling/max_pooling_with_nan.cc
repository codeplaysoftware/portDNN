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
#include <gtest/gtest.h>

#include "portdnn/padding_mode.h"

#include "portdnn/backend/snn_backend.h"

#include "portdnn/helpers/scope_exit.h"

#include "portdnn/pooling/launch.h"
#include "portdnn/pooling/operators.h"
#include "portdnn/pooling/params.h"
#include "portdnn/pooling/sizes.h"

#include "portdnn/status.h"

#include "test/backend/backend_test_fixture.h"
#include "test/pooling/pooling_fixture.h"
#include "test/types/kernel_data_types.h"

#include <stddef.h>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

template <typename DType>
struct MaxPoolingWithNan
    : public BackendTestFixture<sycldnn::backend::SNNBackend> {
  using DataType = DType;

  template <template <typename> class Op>
  void test_forward(std::vector<DataType> input, std::vector<DataType> exp,
                    sycldnn::pooling::PoolingParams const& params) {
    using Direction = sycldnn::pooling::Forward;
    auto in_size = input.size();
    auto out_size = exp.size();
    std::vector<DataType> output(out_size);

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    auto inp_gpu = provider.get_initialised_device_memory(in_size, input);
    auto out_gpu = provider.get_initialised_device_memory(out_size, output);
    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(inp_gpu);
      provider.deallocate_ptr(out_gpu);
    };

    auto status = sycldnn::pooling::launch<DataType, Op, Direction>(
        inp_gpu, out_gpu, params, backend);

    ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
    status.event.wait_and_throw();

    provider.copy_device_data_to_host(out_size, out_gpu, output);

    for (size_t i = 0; i < exp.size(); ++i) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      if (std::isnan(exp[i])) {
        EXPECT_TRUE(std::isnan(output[i]));
      } else if (std::is_same<DataType, double>::value) {
        EXPECT_DOUBLE_EQ(exp[i], output[i]);
      } else {
        EXPECT_FLOAT_EQ(exp[i], output[i]);
      }
    }
  }

  template <template <typename> class Op>
  void test_backprop(std::vector<DataType> const& input_data,
                     std::vector<DataType> const& input_backprop,
                     std::vector<DataType> const& exp,
                     sycldnn::pooling::PoolingParams const& params) {
    auto pooling_size =
        sycldnn::pooling::get_sizes<sycldnn::pooling::Forward>(params);
    auto in_size = pooling_size.input_size;
    auto out_size = pooling_size.output_size;

    std::vector<DataType> output_data(out_size);
    std::vector<DataType> output_backprop(in_size);
    ASSERT_EQ(in_size, input_data.size());
    ASSERT_EQ(out_size, input_backprop.size());

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    auto inp_data_gpu =
        provider.get_initialised_device_memory(in_size, input_data);
    auto out_data_gpu =
        provider.get_initialised_device_memory(out_size, output_data);
    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(inp_data_gpu);
      provider.deallocate_ptr(out_data_gpu);
    };

    auto fwd_status =
        sycldnn::pooling::launch<DataType, Op, sycldnn::pooling::Forward>(
            inp_data_gpu, out_data_gpu, params, backend);
    ASSERT_EQ(sycldnn::StatusCode::OK, fwd_status.status);

    auto inp_backprop_gpu =
        provider.get_initialised_device_memory(out_size, input_backprop);
    auto out_backprop_gpu =
        provider.get_initialised_device_memory(in_size, output_backprop);
    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(inp_backprop_gpu);
      provider.deallocate_ptr(out_backprop_gpu);
    };

    fwd_status.event.wait_and_throw();

    auto back_status =
        sycldnn::pooling::launch<DataType, Op, sycldnn::pooling::Backpropagate>(
            inp_data_gpu, out_data_gpu, inp_backprop_gpu, out_backprop_gpu,
            params, backend);
    ASSERT_EQ(sycldnn::StatusCode::OK, back_status.status);

    back_status.event.wait_and_throw();

    provider.copy_device_data_to_host(in_size, out_backprop_gpu,
                                      output_backprop);

    for (size_t i = 0; i < exp.size(); ++i) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      if (std::isnan(exp[i])) {
        EXPECT_TRUE(std::isnan(output_backprop[i]));
      } else if (std::is_same<DataType, double>::value) {
        EXPECT_DOUBLE_EQ(exp[i], output_backprop[i]);
      } else {
        EXPECT_FLOAT_EQ(exp[i], output_backprop[i]);
      }
    }
  }
};

TYPED_TEST_SUITE(MaxPoolingWithNan, sycldnn::types::GTestKernelDataTypes);

TYPED_TEST(MaxPoolingWithNan, ForwardNan1x1) {
  using DataType = typename TestFixture::DataType;
  const DataType nan = std::numeric_limits<DataType>::quiet_NaN();
  const std::vector<DataType> input = {nan};
  const std::vector<DataType> exp_out = {nan};
  const std::array<int, 4> in_shape = {{1, 1, 1, 1}};
  const auto padding = sycldnn::PaddingMode::VALID;
  const auto params = getPoolingParams<1, 1>(in_shape, padding);
  this->template test_forward<sycldnn::pooling::MaxWithNan>(input, exp_out,
                                                            params);
}

TYPED_TEST(MaxPoolingWithNan, ForwardNan2x2) {
  using DataType = typename TestFixture::DataType;
  const DataType nan = std::numeric_limits<DataType>::quiet_NaN();
  const std::vector<DataType> input = {nan, 2., 3., 4., 5., 6., 7., 8., nan};
  const std::vector<DataType> exp_out = {nan, 6., 8., nan};
  const std::array<int, 4> in_shape = {{1, 3, 3, 1}};
  const auto padding = sycldnn::PaddingMode::VALID;
  const auto params = getPoolingParams<2, 1>(in_shape, padding);
  this->template test_forward<sycldnn::pooling::MaxWithNan>(input, exp_out,
                                                            params);
}

TYPED_TEST(MaxPoolingWithNan, ForwardNoNan2x2) {
  using DataType = typename TestFixture::DataType;
  const DataType nan = std::numeric_limits<DataType>::quiet_NaN();
  const std::vector<DataType> input = {nan, 2., 3., 4., 5., 6., 7., 8., nan};
  const std::vector<DataType> exp_out = {5., 6., 8., 8.};
  const std::array<int, 4> in_shape = {{1, 3, 3, 1}};
  const auto padding = sycldnn::PaddingMode::VALID;
  const auto params = getPoolingParams<2, 1>(in_shape, padding);
  this->template test_forward<sycldnn::pooling::Max>(input, exp_out, params);
}

TYPED_TEST(MaxPoolingWithNan, BackpropNan1x1) {
  using DataType = typename TestFixture::DataType;
  const DataType nan = std::numeric_limits<DataType>::quiet_NaN();
  const std::vector<DataType> input_data = {nan};
  const std::vector<DataType> input_errors = {1.};
  const std::vector<DataType> exp_out = {1.};
  const std::array<int, 4> in_shape = {{1, 1, 1, 1}};
  const auto padding = sycldnn::PaddingMode::VALID;
  const auto params = getPoolingParams<1, 1>(in_shape, padding);
  this->template test_backprop<sycldnn::pooling::MaxWithNan>(
      input_data, input_errors, exp_out, params);
}

TYPED_TEST(MaxPoolingWithNan, BackpropNan2x2) {
  using DataType = typename TestFixture::DataType;
  const DataType nan = std::numeric_limits<DataType>::quiet_NaN();
  const std::vector<DataType> input_data = {nan, 2., 3., 4., 5.,
                                            6.,  7., 8., nan};
  const std::vector<DataType> input_errors = {1., 2., 3., 4.};
  const std::vector<DataType> exp_out = {1., 0., 0., 0., 0., 2., 0., 3., 4.};
  const std::array<int, 4> in_shape = {{1, 3, 3, 1}};
  const auto padding = sycldnn::PaddingMode::VALID;
  const auto params = getPoolingParams<2, 1>(in_shape, padding);
  this->template test_backprop<sycldnn::pooling::MaxWithNan>(
      input_data, input_errors, exp_out, params);
}

// The following tests with an input made up of all NaNs mimic a similar set of
// tests within Tensorflow, which illustrates how the different NaN propagation
// within the max pooling kernels can affect the outputs.
TYPED_TEST(MaxPoolingWithNan, BackpropNoNan2x2) {
  using DataType = typename TestFixture::DataType;
  const DataType nan = std::numeric_limits<DataType>::quiet_NaN();
  const std::vector<DataType> input_data = {nan, 2., 3., 4., 5.,
                                            6.,  7., 8., nan};
  const std::vector<DataType> input_errors = {1., 2., 3., 4.};
  const std::vector<DataType> exp_out = {0., 0., 0., 0., 1., 2., 0., 7., 0.};
  const std::array<int, 4> in_shape = {{1, 3, 3, 1}};
  const auto padding = sycldnn::PaddingMode::VALID;
  const auto params = getPoolingParams<2, 1>(in_shape, padding);
  this->template test_backprop<sycldnn::pooling::Max>(input_data, input_errors,
                                                      exp_out, params);
}

TYPED_TEST(MaxPoolingWithNan, ForwardAllNan) {
  using DataType = typename TestFixture::DataType;
  const DataType nan = std::numeric_limits<DataType>::quiet_NaN();
  const std::vector<DataType> input(16, nan);
  const std::vector<DataType> exp_out(9, nan);
  const std::array<int, 4> in_shape = {{1, 4, 4, 1}};
  const auto padding = sycldnn::PaddingMode::VALID;
  const auto params = getPoolingParams<2, 1>(in_shape, padding);
  this->template test_forward<sycldnn::pooling::MaxWithNan>(input, exp_out,
                                                            params);
}

TYPED_TEST(MaxPoolingWithNan, ForwardAllNoNan) {
  using DataType = typename TestFixture::DataType;
  const DataType nan = std::numeric_limits<DataType>::quiet_NaN();
  const DataType min = std::numeric_limits<DataType>::lowest();
  const std::vector<DataType> input(16, nan);
  const std::vector<DataType> exp_out(9, min);
  const std::array<int, 4> in_shape = {{1, 4, 4, 1}};
  const auto padding = sycldnn::PaddingMode::VALID;
  const auto params = getPoolingParams<2, 1>(in_shape, padding);
  this->template test_forward<sycldnn::pooling::Max>(input, exp_out, params);
}

TYPED_TEST(MaxPoolingWithNan, BackpropNanInputValuesErrors) {
  using DataType = typename TestFixture::DataType;
  const DataType nan = std::numeric_limits<DataType>::quiet_NaN();
  const std::vector<DataType> input_data(16, nan);
  const std::vector<DataType> input_errors = {1., 2., 3., 4., 5.,
                                              6,  7., 8., 9.};
  const std::vector<DataType> exp_out = {1., 2., 3., 0., 4., 5., 6., 0.,
                                         7., 8., 9., 0., 0., 0., 0., 0.};
  const std::array<int, 4> in_shape = {{1, 4, 4, 1}};
  const auto padding = sycldnn::PaddingMode::VALID;
  const auto params = getPoolingParams<2, 1>(in_shape, padding);
  this->template test_backprop<sycldnn::pooling::MaxWithNan>(
      input_data, input_errors, exp_out, params);
}

TYPED_TEST(MaxPoolingWithNan, BackpropNoNanInputValuesErrors) {
  using DataType = typename TestFixture::DataType;
  const DataType nan = std::numeric_limits<DataType>::quiet_NaN();
  const std::vector<DataType> input_data(16, nan);
  const std::vector<DataType> input_errors = {1., 2., 3., 4., 5.,
                                              6,  7., 8., 9.};
  const std::vector<DataType> exp_out = {0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0.};
  const std::array<int, 4> in_shape = {{1, 4, 4, 1}};
  const auto padding = sycldnn::PaddingMode::VALID;
  const auto params = getPoolingParams<2, 1>(in_shape, padding);
  this->template test_backprop<sycldnn::pooling::Max>(input_data, input_errors,
                                                      exp_out, params);
}

TYPED_TEST(MaxPoolingWithNan, BackpropNanInputValuesNans) {
  using DataType = typename TestFixture::DataType;
  const DataType nan = std::numeric_limits<DataType>::quiet_NaN();
  const std::vector<DataType> input_data(16, nan);
  const std::vector<DataType> input_errors = {nan, 2., 3., 4., nan,
                                              6,   7., 8., nan};
  const std::vector<DataType> exp_out = {nan, 2., 3.,  0., 4., nan, 6., 0.,
                                         7.,  8., nan, 0., 0., 0.,  0., 0.};
  const std::array<int, 4> in_shape = {{1, 4, 4, 1}};
  const auto padding = sycldnn::PaddingMode::VALID;
  const auto params = getPoolingParams<2, 1>(in_shape, padding);
  this->template test_backprop<sycldnn::pooling::MaxWithNan>(
      input_data, input_errors, exp_out, params);
}

TYPED_TEST(MaxPoolingWithNan, BackpropNoNanInputValuesNans) {
  using DataType = typename TestFixture::DataType;
  const DataType nan = std::numeric_limits<DataType>::quiet_NaN();
  const std::vector<DataType> input_data(16, nan);
  const std::vector<DataType> input_errors = {nan, 2., 3., 4., nan,
                                              6,   7., 8., nan};
  const std::vector<DataType> exp_out = {0., 0., 0., 0., 0., 0., 0., 0.,
                                         0., 0., 0., 0., 0., 0., 0., 0.};
  const std::array<int, 4> in_shape = {{1, 4, 4, 1}};
  const auto padding = sycldnn::PaddingMode::VALID;
  const auto params = getPoolingParams<2, 1>(in_shape, padding);
  this->template test_backprop<sycldnn::pooling::Max>(input_data, input_errors,
                                                      exp_out, params);
}
