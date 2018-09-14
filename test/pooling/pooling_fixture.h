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

#ifndef SYCLDNN_TEST_POOLING_POOLING_FIXTURE_H_
#define SYCLDNN_TEST_POOLING_POOLING_FIXTURE_H_

#include <gtest/gtest.h>

#include <cassert>
#include <unsupported/Eigen/CXX11/Tensor>

#include "sycldnn/padding_mode.h"

#include "sycldnn/backend/eigen_backend.h"

#include "sycldnn/pooling/add_padding_to_params.h"
#include "sycldnn/pooling/launch.h"
#include "sycldnn/pooling/operators.h"
#include "sycldnn/pooling/params.h"
#include "sycldnn/pooling/sizes.h"

#include "test/gen/eigen_generated_test_fixture.h"
#include "test/gen/iota_initialised_data.h"

#include <vector>

template <typename DType>
struct PoolingFixture
    : public EigenGeneratedTestFixture<DType, sycldnn::backend::EigenBackend> {
  using DataType = DType;

  template <typename Direction, template <typename T> class Op>
  void test_pool(std::vector<DataType> exp,
                 sycldnn::pooling::PoolingParams const& params,
                 DataType max_val = DataType{0}) {
    auto pooling_size = sycldnn::pooling::get_sizes<Direction>(params);
    auto in_size = pooling_size.input_size;
    auto out_size = pooling_size.output_size;
    std::vector<DataType> input = iota_initialised_data(in_size, max_val);
    std::vector<DataType> output(out_size);

    auto inp_gpu = this->get_initialised_device_memory(in_size, input);
    auto out_gpu = this->get_initialised_device_memory(out_size, output);

    auto status = sycldnn::pooling::launch<DataType, Op, Direction>(
        inp_gpu, out_gpu, params, this->backend_);

    ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
    status.event.wait();

    this->copy_device_data_to_host(out_size, out_gpu, output);
    this->deallocate_ptr(inp_gpu);
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

template <int Window, int Stride>
inline sycldnn::pooling::PoolingParams getPoolingParams(
    std::array<int, 4> in_shape, sycldnn::PaddingMode pad) {
  sycldnn::pooling::PoolingParams ret;
  ret.in_rows = in_shape[1];
  ret.in_cols = in_shape[2];
  ret.window_rows = Window;
  ret.window_cols = Window;
  ret.stride_rows = Stride;
  ret.stride_cols = Stride;
  ret.batch = in_shape[0];
  ret.channels = in_shape[3];

  ret = sycldnn::pooling::helpers::add_padding_to(ret, pad);
  return ret;
}

#endif  // SYCLDNN_TEST_POOLING_POOLING_FIXTURE_H_
