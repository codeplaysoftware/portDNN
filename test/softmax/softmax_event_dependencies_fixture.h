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

#ifndef PORTDNN_TEST_SOFTMAX_SOFTMAX_FIXTURE_H_
#define PORTDNN_TEST_SOFTMAX_SOFTMAX_FIXTURE_H_

#include <gtest/gtest.h>

#include "portdnn/helpers/scope_exit.h"

#include "portdnn/softmax/direction.h"
#include "portdnn/softmax/launch.h"
#include "portdnn/softmax/params.h"

#include "test/backend/backend_test_fixture.h"
#include "test/gen/iota_initialised_data.h"
#include "test/helpers/dependency_check.h"
#include "test/helpers/float_comparison.h"
#include "test/helpers/transpose.h"

inline sycldnn::softmax::SoftmaxParams getSoftmaxParams(
    std::array<int, 4> in_shape) {
  sycldnn::softmax::SoftmaxParams params;
  params.channels = in_shape[3];
  params.batch = in_shape[0];
  params.rows = in_shape[1];
  params.cols = in_shape[2];
  params.input_format = sycldnn::DataFormat::NHWC;
  return params;
}

template <class T>
const std::vector<T>& transposeInput(
    sycldnn::softmax::SoftmaxParams const& params, std::vector<T>& trInputData,
    const std::vector<T>& inputData) {
  if (params.input_format == sycldnn::DataFormat::NCHW) {
    transpose(trInputData, inputData, params.batch, params.rows * params.cols,
              params.channels);
    return trInputData;
  }
  return inputData;
}

template <class T>
const std::vector<T>& transposeOutput(
    sycldnn::softmax::SoftmaxParams const& params, std::vector<T>& trOutputData,
    const std::vector<T>& outputData) {
  if (params.input_format == sycldnn::DataFormat::NCHW) {
    transpose(trOutputData, outputData, params.batch, params.channels,
              params.rows * params.cols);
    return trOutputData;
  }
  return outputData;
}

template <typename Pair, typename Direction>
struct SoftmaxEventFixture;

template <typename Pair>
struct SoftmaxEventFixture<Pair, sycldnn::softmax::Forward>
    : public BackendTestFixture<sycldnn::backend::SNNUSMBackend> {
  using DataType = typename Pair::FirstType;
  static constexpr sycldnn::DataFormat INPUT_FORMAT =
      Pair::SecondType::input_layout;

  void test_softmax(sycldnn::softmax::SoftmaxParams params,
                    DataType max_val = static_cast<DataType>(0)) {
    ASSERT_EQ(params.input_format, sycldnn::DataFormat::NHWC)
        << "Tests should be written for the NHWC layout. The input layout is "
           "set from the fixture type.";
    params.input_format = INPUT_FORMAT;
    auto size = params.batch * params.rows * params.cols * params.channels;
    auto workspace_size = params.batch * params.rows * params.cols;

    std::vector<DataType> inputData =
        iota_initialised_data<DataType>(size, max_val);
    std::vector<DataType> outputData(size);
    std::vector<DataType> workspace(workspace_size);

    std::vector<DataType> trInputData;
    const std::vector<DataType>& input =
        transposeInput(params, trInputData, inputData);

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    auto inp_gpu = provider.get_initialised_device_memory(size, input);
    auto workspace_gpu =
        provider.get_initialised_device_memory(workspace_size, workspace);
    auto out_gpu = provider.get_initialised_device_memory(size, outputData);
    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(inp_gpu);
      provider.deallocate_ptr(out_gpu);
      provider.deallocate_ptr(workspace_gpu);
    };

    dependency_test_params dep_test_params;
    cl::sycl::event dependee_e = create_event(backend, dep_test_params);

    auto status = sycldnn::softmax::launch<DataType, sycldnn::softmax::Forward>(
        inp_gpu, workspace_gpu, out_gpu, params, backend,
        std::vector<cl::sycl::event>{dependee_e});

    ASSERT_EQ(sycldnn::StatusCode::OK, status.status);

    check_dependency(dependee_e, status.event, backend, dep_test_params);
  }
};

template <typename Pair>
struct SoftmaxEventFixture<Pair, sycldnn::softmax::Gradient>
    : public BackendTestFixture<sycldnn::backend::SNNUSMBackend> {
  using DataType = typename Pair::FirstType;
  static constexpr sycldnn::DataFormat INPUT_FORMAT =
      Pair::SecondType::input_layout;

  void test_softmax(sycldnn::softmax::SoftmaxParams params,
                    DataType max_val = static_cast<DataType>(0)) {
    ASSERT_EQ(params.input_format, sycldnn::DataFormat::NHWC)
        << "Tests should be written for the NHWC layout. The input layout is "
           "set from the fixture type.";
    params.input_format = INPUT_FORMAT;
    auto size = params.batch * params.rows * params.cols * params.channels;
    auto workspace_size = params.batch * params.rows * params.cols;

    std::vector<DataType> inputData =
        iota_initialised_data<DataType>(size, max_val);
    std::vector<DataType> outputData(size);
    std::vector<DataType> workspace_fwd(workspace_size);
    std::vector<DataType> workspace_grad(size);

    std::vector<DataType> trInputData;
    const std::vector<DataType>& input =
        transposeInput(params, trInputData, inputData);

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    auto inp_gpu = provider.get_initialised_device_memory(size, input);
    auto workspace_fwd_gpu =
        provider.get_initialised_device_memory(workspace_size, workspace_fwd);
    auto workspace_grad_gpu =
        provider.get_initialised_device_memory(size, workspace_grad);
    auto out_fwd_gpu = provider.get_initialised_device_memory(size, outputData);
    auto out_grad_gpu =
        provider.get_initialised_device_memory(size, outputData);
    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(inp_gpu);
      provider.deallocate_ptr(out_fwd_gpu);
      provider.deallocate_ptr(out_grad_gpu);
      provider.deallocate_ptr(workspace_fwd_gpu);
      provider.deallocate_ptr(workspace_grad_gpu);
    };

    auto status = sycldnn::softmax::launch<DataType, sycldnn::softmax::Forward>(
        inp_gpu, workspace_fwd_gpu, out_fwd_gpu, params, backend);

    status.event.wait_and_throw();

    dependency_test_params dep_test_params;
    cl::sycl::event dependee_e = create_event(backend, dep_test_params);

    status = sycldnn::softmax::launch<DataType, sycldnn::softmax::Gradient>(
        out_fwd_gpu, inp_gpu, workspace_grad_gpu, out_grad_gpu, params, backend,
        {dependee_e});

    ASSERT_EQ(sycldnn::StatusCode::OK, status.status);

    check_dependency(dependee_e, status.event, backend, dep_test_params);
  }
};

#endif  // PORTDNN_TEST_SOFTMAX_SOFTMAX_FIXTURE_H_
