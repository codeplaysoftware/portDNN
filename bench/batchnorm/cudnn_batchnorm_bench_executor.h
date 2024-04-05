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

#pragma once

#include "src/backend/snn_backend_provider.h"
#include "sycldnn/backend/snn_backend.h"

#include "sycldnn/batchnorm/direction.h"
#include "sycldnn/batchnorm/params.h"

#include "sycldnn/helpers/scope_exit.h"

#include "sycldnn/status.h"

#include "base_batchnorm_fixture.h"
#include "benchmark_config.h"
#include "benchmark_params.h"

#include "bench/batchnorm/snn_batchnorm_executor.h"

#include "bench/common/cudnn_common.h"
#include "bench/common/verification_utils.h"

#include "bench/fixture/add_datatype_info.h"
#include "bench/fixture/base_bench_executor.h"
#include "bench/fixture/statistic.h"
#include "bench/fixture/string_reporter.h"
#include "bench/fixture/typenames.h"

#include <benchmark/benchmark.h>

#include <cuda.h>
#include <cudnn.h>
#include <cudnn_cnn_infer.h>
#include <cudnn_ops_infer.h>

#include <array>
#include <functional>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>

namespace sycldnn {
namespace bench {

template <typename Benchmark, typename DataType, typename Direction>
struct CUDNNBatchnormExecutor : public BaseBenchExecutor {
 private:
  using State = ::benchmark::State;
  using Params = batchnorm::BatchNormParams;
  static constexpr bool IsForwardInference =
      std::is_same_v<Direction, sycldnn::batchnorm::ForwardInference>;
  static constexpr bool IsForwardTraining =
      std::is_same_v<Direction, sycldnn::batchnorm::ForwardTraining>;
  static constexpr bool IsGradient =
      std::is_same_v<Direction, sycldnn::batchnorm::Gradient>;

  /** Get a reference to the underlying benchmark fixture. */
  Benchmark& underlying_benchmark() { return static_cast<Benchmark&>(*this); }

  void validate_results(void* cudnn_gpu, void* running_mean_cudnn_gpu,
                        void* running_var_cudnn_gpu, void* gamma_grad_cudnn_gpu,
                        void* beta_grad_cudnn_gpu,
                        std::vector<std::vector<DataType>> const& host_inputs,
                        Params const& params, DataType tolerance = 1e-5) {
    using Executor =
        SNNBatchnormExecutor<DataType, Direction, backend::SNNBackend, false>;

    Executor snn_batchnorm(host_inputs[0], host_inputs[1], host_inputs[2],
                           host_inputs[3], host_inputs[4], params);

    SNN_ON_SCOPE_EXIT { snn_batchnorm.teardown(); };

    // Allocate SNN device memory
    snn_batchnorm.setup();

    // Execute SNN batchnorm
    auto status = snn_batchnorm.execute();
    if (status.status != sycldnn::StatusCode::OK) {
      throw std::runtime_error(
          "Unexpected issue executing SNN batchnorm for CuDNN benchmark "
          "validation.");
    }

    status.event.wait_and_throw();

    const auto output_size =
        params.batch * params.rows * params.cols * params.channels;

    if constexpr (IsForwardInference || IsForwardTraining) {
      std::vector<DataType> snn_output(output_size);
      std::vector<DataType> cudnn_output(output_size);

      // Copy SNN results to host
      snn_batchnorm.copy_forward_output_to_host(snn_output);
      // Copy CuDNN results to host
      cuda_check(cudaMemcpy, cudnn_output.data(), cudnn_gpu,
                 sizeof(DataType) * output_size, cudaMemcpyDeviceToHost);
      cuda_check(cudaDeviceSynchronize);

      // Assert that results are equal.
      if (!internal::compare_results(snn_output, cudnn_output, tolerance)) {
        throw std::runtime_error(
            "CuDNN results do not match SNN results, exiting benchmark.");
      }

      if constexpr (IsForwardTraining) {
        std::vector<DataType> snn_running_mean(params.channels);
        std::vector<DataType> snn_running_var(params.channels);
        snn_batchnorm.copy_running_mean_var_to_host(snn_running_mean,
                                                    snn_running_var);

        std::vector<DataType> cudnn_running_mean(params.channels);
        std::vector<DataType> cudnn_running_var(params.channels);
        cuda_check(cudaMemcpy, cudnn_running_mean.data(),
                   running_mean_cudnn_gpu, sizeof(DataType) * params.channels,
                   cudaMemcpyDeviceToHost);
        cuda_check(cudaMemcpy, cudnn_running_var.data(), running_var_cudnn_gpu,
                   sizeof(DataType) * params.channels, cudaMemcpyDeviceToHost);
        cuda_check(cudaDeviceSynchronize);

        // cuDNN computes unbiased variance (i.e. in the variance formula, it
        // divides by n-1 rather than n), therefore convert the cudnn output to
        // population variance to compare it to portDNN
        const auto elems_per_channel = params.batch * params.rows * params.cols;
        std::transform(
            cudnn_running_var.begin(), cudnn_running_var.end(),
            cudnn_running_var.begin(), [elems_per_channel](auto& element) {
              return element * (elems_per_channel - 1) / elems_per_channel;
            });

        if (!internal::compare_results(snn_running_mean, cudnn_running_mean,
                                       tolerance)) {
          throw std::runtime_error(
              "CuDNN results for running mean do not match SNN results, "
              "exiting benchmark.");
        }
        if (!internal::compare_results(snn_running_var, cudnn_running_var,
                                       tolerance)) {
          throw std::runtime_error(
              "CuDNN results for running variance do not match SNN results, "
              "exiting benchmark.");
        }
      }
    } else {  // Gradient
      std::vector<DataType> in_grad_snn(output_size),
          gamma_grad_snn(params.channels), beta_grad_snn(params.channels);
      std::vector<DataType> in_grad_cudnn(output_size),
          gamma_grad_cudnn(params.channels), beta_grad_cudnn(params.channels);

      snn_batchnorm.copy_gradient_outputs_to_host(in_grad_snn, gamma_grad_snn,
                                                  beta_grad_snn);

      cuda_check(cudaMemcpy, in_grad_cudnn.data(), cudnn_gpu,
                 sizeof(DataType) * output_size, cudaMemcpyDeviceToHost);
      cuda_check(cudaMemcpy, gamma_grad_cudnn.data(), gamma_grad_cudnn_gpu,
                 sizeof(DataType) * params.channels, cudaMemcpyDeviceToHost);
      cuda_check(cudaMemcpy, beta_grad_cudnn.data(), beta_grad_cudnn_gpu,
                 sizeof(DataType) * params.channels, cudaMemcpyDeviceToHost);
      cuda_check(cudaDeviceSynchronize);

      if (!internal::compare_results(in_grad_snn, in_grad_snn, tolerance)) {
        throw std::runtime_error(
            "CuDNN results for input gradient do not match SNN results, "
            "exiting benchmark.");
      }
      if (!internal::compare_results(gamma_grad_snn, gamma_grad_cudnn,
                                     tolerance)) {
        throw std::runtime_error(
            "CuDNN results for gamma gradient do not match SNN results, "
            "exiting benchmark.");
      }
      if (!internal::compare_results(beta_grad_snn, beta_grad_cudnn,
                                     tolerance)) {
        throw std::runtime_error(
            "CuDNN results for beta gradient do not match SNN results, "
            "exiting benchmark.");
      }
    }
  }

 protected:
  cudnnHandle_t handle_;

 public:
  /** Execute a batchnorm benchmark with the given parameters. */
  void execute(State& state, Params const& params) {
    using TensorShape = std::array<int32_t, 4>;

    auto alpha = 1.f;
    auto beta = 0.f;
    auto cudnn_tensor_format = get_cudnn_tensor_format(params.input_format);
    // cuDNN BatchNorm always expects NCHW
    TensorShape layer_shape = {params.batch, params.channels, params.rows,
                               params.cols};
    cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL;  // 1, C, 1, 1

    cudnnTensorDescriptor_t inp_out_desc;
    cudnnTensorDescriptor_t scale_bias_desc;
    cudnn_check(cudnnCreateTensorDescriptor, &inp_out_desc);
    cudnn_check(cudnnCreateTensorDescriptor, &scale_bias_desc);
    cudnn_check(cudnnSetTensor4dDescriptor, inp_out_desc, cudnn_tensor_format,
                CUDNN_DATA_FLOAT, layer_shape[0], layer_shape[1],
                layer_shape[2], layer_shape[3]);
    cudnn_check(cudnnDeriveBNTensorDescriptor, scale_bias_desc, inp_out_desc,
                mode);

    size_t input_size =
        params.batch * params.rows * params.cols * params.channels;
    size_t scale_bias_size = params.channels;

    const DataType max_val = 2.0;
    const auto inp_host = internal::iota_initialised_data(input_size, max_val);
    const auto scale_host =
        internal::iota_initialised_data(params.channels, max_val);
    const auto bias_host =
        internal::iota_initialised_data(params.channels, max_val);

    std::vector<DataType> estimated_mean_host, estimated_variance_host;
    if constexpr (IsForwardInference || IsGradient) {
      estimated_mean_host =
          internal::iota_initialised_data(params.channels, max_val);
      estimated_variance_host =
          internal::iota_initialised_data(params.channels, max_val);
    } else {
      estimated_mean_host.resize(params.channels);
      estimated_variance_host.resize(params.channels);
    }

    DataType* inp_gpu = nullptr;
    DataType* out_gpu = nullptr;
    DataType* scale_gpu = nullptr;
    DataType* bias_gpu = nullptr;
    DataType* estimated_mean_gpu = nullptr;
    DataType* estimated_variance_gpu = nullptr;
    DataType* running_mean_gpu = nullptr;
    DataType* running_var_gpu = nullptr;
    DataType* inp_grad_gpu = nullptr;
    DataType* out_grad_gpu = nullptr;
    DataType* scale_grad_gpu = nullptr;
    DataType* bias_grad_gpu = nullptr;

    const auto input_size_bytes = input_size * sizeof(DataType);
    const auto scale_bias_size_bytes = scale_bias_size * sizeof(DataType);

    // Forward inference
    // Inputs: input, gamma, beta, mean, var
    // Outputs: out
    cuda_check(cudaMalloc<DataType>, &inp_gpu, input_size_bytes);
    cuda_check(cudaMalloc<DataType>, &out_gpu, input_size_bytes);
    cuda_check(cudaMalloc<DataType>, &scale_gpu, scale_bias_size_bytes);
    cuda_check(cudaMalloc<DataType>, &bias_gpu, scale_bias_size_bytes);
    cuda_check(cudaMalloc<DataType>, &estimated_mean_gpu,
               scale_bias_size_bytes);
    cuda_check(cudaMalloc<DataType>, &estimated_variance_gpu,
               scale_bias_size_bytes);

    cuda_check(cudaMemcpy, inp_gpu, inp_host.data(), input_size_bytes,
               cudaMemcpyHostToDevice);
    cuda_check(cudaMemcpy, scale_gpu, scale_host.data(), scale_bias_size_bytes,
               cudaMemcpyHostToDevice);
    cuda_check(cudaMemcpy, bias_gpu, bias_host.data(), scale_bias_size_bytes,
               cudaMemcpyHostToDevice);
    cuda_check(cudaMemcpy, estimated_mean_gpu, estimated_mean_host.data(),
               scale_bias_size_bytes, cudaMemcpyHostToDevice);
    cuda_check(cudaMemcpy, estimated_variance_gpu,
               estimated_variance_host.data(), scale_bias_size_bytes,
               cudaMemcpyHostToDevice);

    // Forward training
    // Inputs: input, gamma, beta
    // Outputs: running mean, running var, cache mean, cache var
    if constexpr (IsForwardTraining) {
      cuda_check(cudaMalloc<DataType>, &running_mean_gpu,
                 scale_bias_size_bytes);
      cuda_check(cudaMalloc<DataType>, &running_var_gpu, scale_bias_size_bytes);
      cuda_check(cudaMemset, running_mean_gpu, 0, scale_bias_size_bytes);
      cuda_check(cudaMemset, running_var_gpu, 0, scale_bias_size_bytes);
    }

    // Gradient
    // Inputs: out grad, input, gamma, cache mean, cache var
    // Outputs: in grad, gamma grad, beta grad
    if constexpr (IsGradient) {
      cuda_check(cudaMalloc<DataType>, &inp_grad_gpu, input_size_bytes);
      cuda_check(cudaMalloc<DataType>, &out_grad_gpu, input_size_bytes);
      cuda_check(cudaMalloc<DataType>, &scale_grad_gpu, scale_bias_size_bytes);
      cuda_check(cudaMalloc<DataType>, &bias_grad_gpu, scale_bias_size_bytes);
      cuda_check(cudaMemset, inp_grad_gpu, 0, input_size_bytes);
      cuda_check(cudaMemcpy, out_grad_gpu, inp_host.data(), input_size_bytes,
                 cudaMemcpyHostToDevice);
      cuda_check(cudaMemset, scale_grad_gpu, 0, scale_bias_size_bytes);
      cuda_check(cudaMemset, bias_grad_gpu, 0, scale_bias_size_bytes);
    }
    cuda_check(cudaDeviceSynchronize);

    SNN_ON_SCOPE_EXIT {
      cudnnDestroyTensorDescriptor(inp_out_desc);
      cudnnDestroyTensorDescriptor(scale_bias_desc);
      cudaFree(inp_gpu);
      cudaFree(out_gpu);
      cudaFree(scale_gpu);
      cudaFree(bias_gpu);
      cudaFree(estimated_mean_gpu);
      cudaFree(estimated_variance_gpu);
      if (running_mean_gpu && IsForwardTraining) {
        cudaFree(running_mean_gpu);
      }
      if (running_var_gpu && IsForwardTraining) {
        cudaFree(running_var_gpu);
      }
      if constexpr (IsGradient) {
        cudaFree(inp_grad_gpu);
        cudaFree(out_grad_gpu);
        cudaFree(scale_grad_gpu);
        cudaFree(bias_grad_gpu);
      }
    };

    float constexpr tolerance = 1e-4;
    auto snn_inputs =
        std::vector({inp_host, scale_host, bias_host, estimated_mean_host,
                     estimated_variance_host});
    if constexpr (IsForwardInference) {
      cudnn_check(cudnnBatchNormalizationForwardInference, this->handle_, mode,
                  &alpha, &beta, inp_out_desc, inp_gpu, inp_out_desc, out_gpu,
                  scale_bias_desc, scale_gpu, bias_gpu, estimated_mean_gpu,
                  estimated_variance_gpu, params.epsilon);
      cuda_check(cudaDeviceSynchronize);
      this->validate_results(out_gpu, nullptr, nullptr, nullptr, nullptr,
                             snn_inputs, params, tolerance);

      for (auto _ : state) {
        this->start_timing();

        cudnn_check(cudnnBatchNormalizationForwardInference, this->handle_,
                    mode, &alpha, &beta, inp_out_desc, inp_gpu, inp_out_desc,
                    out_gpu, scale_bias_desc, scale_gpu, bias_gpu,
                    estimated_mean_gpu, estimated_variance_gpu, params.epsilon);
        cuda_check(cudaDeviceSynchronize);

        this->end_timing();
        this->set_iteration_time(state);
      }
    } else if constexpr (IsForwardTraining) {
      cudnn_check(cudnnBatchNormalizationForwardTraining, this->handle_, mode,
                  &alpha, &beta, inp_out_desc, inp_gpu, inp_out_desc, out_gpu,
                  scale_bias_desc, scale_gpu, bias_gpu, (1.f - params.momentum),
                  running_mean_gpu, running_var_gpu, params.epsilon,
                  estimated_mean_gpu, estimated_variance_gpu);
      cuda_check(cudaDeviceSynchronize);
      this->validate_results(out_gpu, running_mean_gpu, running_var_gpu,
                             nullptr, nullptr, snn_inputs, params, tolerance);

      for (auto _ : state) {
        this->start_timing();

        cudnn_check(cudnnBatchNormalizationForwardTraining, this->handle_, mode,
                    &alpha, &beta, inp_out_desc, inp_gpu, inp_out_desc, out_gpu,
                    scale_bias_desc, scale_gpu, bias_gpu, params.momentum,
                    running_mean_gpu, running_var_gpu, params.epsilon,
                    estimated_mean_gpu, estimated_variance_gpu);
        cuda_check(cudaDeviceSynchronize);

        this->end_timing();
        this->set_iteration_time(state);
      }
    } else {  // Gradient
      cudnn_check(cudnnBatchNormalizationBackward, this->handle_, mode, &alpha,
                  &beta, &alpha, &beta, inp_out_desc, inp_gpu, inp_out_desc,
                  out_grad_gpu, inp_out_desc, inp_grad_gpu, scale_bias_desc,
                  scale_gpu, scale_grad_gpu, bias_grad_gpu, params.epsilon,
                  estimated_mean_gpu, estimated_variance_gpu);
      cuda_check(cudaDeviceSynchronize);
      this->validate_results(inp_grad_gpu, nullptr, nullptr, scale_grad_gpu,
                             bias_grad_gpu, snn_inputs, params, tolerance);

      for (auto _ : state) {
        this->start_timing();

        cudnn_check(cudnnBatchNormalizationBackward, this->handle_, mode,
                    &alpha, &beta, &alpha, &beta, inp_out_desc, inp_gpu,
                    inp_out_desc, out_grad_gpu, inp_out_desc, inp_grad_gpu,
                    scale_bias_desc, scale_gpu, scale_grad_gpu, bias_grad_gpu,
                    params.epsilon, estimated_mean_gpu, estimated_variance_gpu);
        cuda_check(cudaDeviceSynchronize);

        this->end_timing();
        this->set_iteration_time(state);
      }
    }

    auto& benchmark = underlying_benchmark();

    benchmark.set_items_processed(state, params);
    benchmark.add_param_counters(state, params);
    benchmark.template add_bandwidth_counters<float>(state, params);

    this->finish_benchmark(state);
  }
};

}  // namespace bench
}  // namespace sycldnn

extern const char* commit_date;
extern const char* commit_hash;

template <typename DataType, typename DataLayout, typename Direction>
class CUDNNBatchnormBenchmark
    : public sycldnn::bench::CUDNNBatchnormExecutor<
          CUDNNBatchnormBenchmark<DataType, DataLayout, Direction>, DataType,
          Direction>,
      public sycldnn::bench::StringReporter,
      public BaseBatchnormBenchmark {
  using State = benchmark::State;
  cudaDeviceProp properties_;
  int version_;

 protected:
  void run(State& state) {
    auto params = benchmark_params::deserialize(state);
    params.input_format = DataLayout::input_layout;
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::MaxStatistic{}});
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::MinStatistic{}});
    this->add_statistic(std::unique_ptr<sycldnn::bench::Statistic>{
        new sycldnn::bench::StdDevStatistic{}});
    try {
      this->execute(state, params);
    } catch (const std::runtime_error& e) {
      state.SkipWithError(e.what());
    }

    sycldnn::bench::datatype_info::add_datatype_info<DataType>(*this);

    this->add_to_label("@library", "SYCL-DNN");
    this->add_to_label("@backend", "cuDNN");
    this->add_to_label("@layout", sycldnn::bench::TypeName<DataLayout>::name);
    this->add_to_label("@direction",
                       std::is_same_v<Direction, sycldnn::batchnorm::Gradient>
                           ? "Gradient"
                           : "Forward");
    this->add_to_label(
        "@is_training",
        std::is_same_v<Direction, sycldnn::batchnorm::ForwardInference>
            ? "false"
            : "true");
    this->add_to_label("short_name", "Batchnorm");
    this->add_to_label("git_hash", commit_hash);

    this->add_to_label("vendor_name", "NVIDIA");
    this->add_to_label("device_name", properties_.name);
    this->add_to_label("device_version",
                       std::to_string(version_ / 1000) + "." +
                           std::to_string((version_ % 1000) / 10));
    this->add_to_label("driver_version", "n/a");
    this->set_label(state);
  }

  void set_model(const char* model_name) {
    this->add_to_label("@model_name", model_name);
  }

 public:
  void SetUp(const State&) {
    cuda_check(cudaSetDevice, 0);
    cuda_check(cudaGetDeviceProperties, &this->properties_, 0);
    cuda_check(cudaRuntimeGetVersion, &this->version_);
    cudnn_check(cudnnCreate, &this->handle_);
  }

  void TearDown(const State&) { cudnn_check(cudnnDestroy, this->handle_); }
};

#define BATCHNORM_BENCHMARK(name, ...)                                    \
  BENCHMARK_TEMPLATE_DEFINE_F(CUDNNBatchnormBenchmark, name, __VA_ARGS__) \
  (benchmark::State & state) {                                            \
    this->set_model(get_benchmark_name());                                \
    this->run(state);                                                     \
  }                                                                       \
  BENCHMARK_REGISTER_F(CUDNNBatchnormBenchmark, name)                     \
      ->UseManualTime()                                                   \
      ->Unit(benchmark::kNanosecond)                                      \
      ->Apply(RunForAllParamSets);
