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

#ifndef PORTDNN_TEST_BATCHNORM_FIXTURE_H_
#define PORTDNN_TEST_BATCHNORM_FIXTURE_H_

#include <gtest/gtest.h>

#include "portdnn/helpers/scope_exit.h"

#include "portdnn/batchnorm/direction.h"
#include "portdnn/batchnorm/launch.h"
#include "portdnn/batchnorm/params.h"

#include "test/backend/backend_test_fixture.h"
#include "test/gen/iota_initialised_data.h"
#include "test/helpers/dependency_check.h"
#include "test/helpers/float_comparison.h"
#include "test/helpers/transpose.h"

inline sycldnn::batchnorm::BatchNormParams getBatchNormParams(
    std::array<int, 4> in_shape, bool is_training, float momentum,
    float epsilon) {
  sycldnn::batchnorm::BatchNormParams params;
  params.batch = in_shape[0];
  params.rows = in_shape[1];
  params.cols = in_shape[2];
  params.channels = in_shape[3];
  params.is_training = is_training;
  params.momentum = momentum;
  params.epsilon = epsilon;
  params.input_format = sycldnn::DataFormat::NHWC;
  return params;
}

template <class T>
const std::vector<T>& transposeInput(
    sycldnn::batchnorm::BatchNormParams const& params,
    std::vector<T>& trInputData, const std::vector<T>& inputData) {
  if (params.input_format == sycldnn::DataFormat::NCHW) {
    transpose(trInputData, inputData, params.batch, params.rows * params.cols,
              params.channels);
    return trInputData;
  }
  return inputData;
}

template <class T>
const std::vector<T>& transposeOutput(
    sycldnn::batchnorm::BatchNormParams const& params,
    std::vector<T>& trOutputData, const std::vector<T>& outputData) {
  if (params.input_format == sycldnn::DataFormat::NCHW) {
    transpose(trOutputData, outputData, params.batch, params.channels,
              params.rows * params.cols);
    return trOutputData;
  }
  return outputData;
}

template <typename Triple, typename Direction>
struct BatchNormEventFixture;

template <typename Pair>
struct BatchNormEventFixture<Pair, sycldnn::batchnorm::Forward>
    : public BackendTestFixture<sycldnn::backend::SNNUSMBackend> {
  using DataType = typename Pair::FirstType;
  using Backend = sycldnn::backend::SNNUSMBackend;
  static constexpr sycldnn::DataFormat INPUT_FORMAT =
      Pair::SecondType::input_layout;

  void test_batchnorm(sycldnn::batchnorm::BatchNormParams params,
                      DataType max_input_val, DataType max_beta_val,
                      DataType max_gamma_val, DataType max_input_mean_val,
                      DataType max_input_var_val) {
    ASSERT_EQ(params.input_format, sycldnn::DataFormat::NHWC)
        << "Tests should be written for the NHWC layout. The input layout is "
           "set from the fixture type.";
    params.input_format = INPUT_FORMAT;

    auto size = params.batch * params.rows * params.cols * params.channels;

    std::vector<DataType> inputData =
        iota_initialised_data<DataType>(size, max_input_val);
    std::vector<DataType> beta =
        iota_initialised_data<DataType>(params.channels, max_beta_val);
    std::vector<DataType> gamma =
        iota_initialised_data<DataType>(params.channels, max_gamma_val);
    std::vector<DataType> input_mean =
        iota_initialised_data<DataType>(params.channels, max_input_mean_val);
    std::vector<DataType> input_var =
        iota_initialised_data<DataType>(params.channels, max_input_var_val);
    std::vector<DataType> outputData(size);
    std::vector<DataType>& output = outputData;

    std::vector<DataType> trInputData;
    const std::vector<DataType>& input =
        transposeInput(params, trInputData, inputData);

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    auto inp_gpu = provider.get_initialised_device_memory(size, input);
    auto beta_gpu =
        provider.get_initialised_device_memory(params.channels, beta);
    auto gamma_gpu =
        provider.get_initialised_device_memory(params.channels, gamma);
    auto input_mean_gpu =
        provider.get_initialised_device_memory(params.channels, input_mean);
    auto input_variance_gpu =
        provider.get_initialised_device_memory(params.channels, input_var);
    auto running_mean_gpu =
        provider.get_initialised_device_memory(params.channels, input_mean);
    auto running_variance_gpu =
        provider.get_initialised_device_memory(params.channels, input_var);
    auto out_gpu = provider.get_initialised_device_memory(size, output);
    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(inp_gpu);
      provider.deallocate_ptr(beta_gpu);
      provider.deallocate_ptr(gamma_gpu);
      provider.deallocate_ptr(input_mean_gpu);
      provider.deallocate_ptr(input_variance_gpu);
      provider.deallocate_ptr(running_mean_gpu);
      provider.deallocate_ptr(running_variance_gpu);
      provider.deallocate_ptr(out_gpu);
    };

    dependency_test_params dep_test_params;
    cl::sycl::event dependee_e = create_event(backend, dep_test_params);

    auto status = sycldnn::batchnorm::launch<DataType, Backend,
                                             sycldnn::batchnorm::Forward>(
        inp_gpu, beta_gpu, gamma_gpu, input_mean_gpu, input_variance_gpu,
        running_mean_gpu, running_variance_gpu, out_gpu, params, backend,
        std::vector<cl::sycl::event>{dependee_e});

    ASSERT_EQ(sycldnn::StatusCode::OK, status.status);

    check_dependency(dependee_e, status.event, backend, dep_test_params);
  }
};

template <typename Pair>
struct BatchNormEventFixture<Pair, sycldnn::batchnorm::Gradient>
    : public BackendTestFixture<sycldnn::backend::SNNUSMBackend> {
  using DataType = typename Pair::FirstType;
  using Backend = sycldnn::backend::SNNUSMBackend;
  static constexpr sycldnn::DataFormat INPUT_FORMAT =
      Pair::SecondType::input_layout;

  void test_batchnorm(sycldnn::batchnorm::BatchNormParams params,
                      DataType max_input_val, DataType max_gradient_val,
                      DataType max_gamma_val, DataType max_pop_mean_val,
                      DataType max_pop_var_val) {
    ASSERT_EQ(params.input_format, sycldnn::DataFormat::NHWC)
        << "Tests should be written for the NHWC layout. The input layout is "
           "set from the fixture type.";
    params.input_format = INPUT_FORMAT;

    auto size = params.batch * params.rows * params.cols * params.channels;
    std::cout << "size:" << size << std::endl;

    std::vector<DataType> inputData =
        iota_initialised_data<DataType>(size, max_input_val);
    std::vector<DataType> gradientData =
        iota_initialised_data<DataType>(size, max_gradient_val);
    std::vector<DataType> gamma =
        iota_initialised_data<DataType>(params.channels, max_gamma_val);
    std::vector<DataType> pop_mean =
        iota_initialised_data<DataType>(params.channels, max_pop_mean_val);
    std::vector<DataType> pop_var =
        iota_initialised_data<DataType>(params.channels, max_pop_var_val);
    std::vector<DataType> beta_grad(params.channels);
    std::vector<DataType> gamma_grad(params.channels);
    std::vector<DataType> outputData(size);
    std::vector<DataType>& output = outputData;

    std::vector<DataType> trInputData;
    const std::vector<DataType>& input =
        transposeInput(params, trInputData, inputData);

    std::vector<DataType> trGradientData;
    const std::vector<DataType>& gradient =
        transposeInput(params, trGradientData, gradientData);

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    auto inp_gpu = provider.get_initialised_device_memory(size, input);
    auto gradient_gpu = provider.get_initialised_device_memory(size, gradient);
    auto gamma_gpu =
        provider.get_initialised_device_memory(params.channels, gamma);
    auto pop_mean_gpu =
        provider.get_initialised_device_memory(params.channels, pop_mean);
    auto pop_variance_gpu =
        provider.get_initialised_device_memory(params.channels, pop_var);
    auto beta_grad_gpu =
        provider.get_initialised_device_memory(params.channels, beta_grad);
    auto gamma_grad_gpu =
        provider.get_initialised_device_memory(params.channels, gamma_grad);
    auto out_gpu = provider.get_initialised_device_memory(size, output);

    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(inp_gpu);
      provider.deallocate_ptr(gradient_gpu);
      provider.deallocate_ptr(gamma_gpu);
      provider.deallocate_ptr(pop_mean_gpu);
      provider.deallocate_ptr(pop_variance_gpu);
      provider.deallocate_ptr(beta_grad_gpu);
      provider.deallocate_ptr(gamma_grad_gpu);
      provider.deallocate_ptr(out_gpu);
    };

    dependency_test_params dep_test_params;
    cl::sycl::event dependee_e = create_event(backend, dep_test_params);

    auto status = sycldnn::batchnorm::launch<DataType, Backend,
                                             sycldnn::batchnorm::Gradient>(
        inp_gpu, gradient_gpu, gamma_gpu, pop_mean_gpu, pop_variance_gpu,
        beta_grad_gpu, gamma_grad_gpu, out_gpu, params, backend,
        std::vector<cl::sycl::event>{dependee_e});

    ASSERT_EQ(sycldnn::StatusCode::OK, status.status);

    check_dependency(dependee_e, status.event, backend, dep_test_params);
  }
};

#endif  // PORTDNN_TEST_BATCHNORM_FIXTURE_H_
