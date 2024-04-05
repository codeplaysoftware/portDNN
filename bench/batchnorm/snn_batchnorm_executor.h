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
#ifndef SYCLDNN_BENCH_BATCHNORM_SNN_BATCHNORM_EXECUTOR_H_
#define SYCLDNN_BENCH_BATCHNORM_SNN_BATCHNORM_EXECUTOR_H_

#include <benchmark/benchmark.h>

#include "sycldnn/helpers/handle_exception.h"
#include "sycldnn/helpers/scope_exit.h"

#include "sycldnn/batchnorm/direction.h"
#include "sycldnn/batchnorm/launch.h"

#include "src/backend/snn_backend_provider.h"

#ifdef SNN_ENABLE_USM
#include "src/backend/snn_usm_backend_provider.h"
#endif

namespace sycldnn {
namespace bench {

template <typename DataType, typename Direction, typename Backend,
          bool InOrderQueue>
struct SNNBatchnormExecutor {
 private:
  using State = ::benchmark::State;
  using Params = batchnorm::BatchNormParams;
  using Provider = typename backend::BackendProvider<Backend, InOrderQueue>;
  using Pointer = typename Provider::template Pointer<DataType>;
  static constexpr bool IsForwardInference =
      std::is_same_v<Direction, sycldnn::batchnorm::ForwardInference>;
  static constexpr bool IsForwardTraining =
      std::is_same_v<Direction, sycldnn::batchnorm::ForwardTraining>;
  static constexpr bool IsGradient =
      std::is_same_v<Direction, sycldnn::batchnorm::Gradient>;

  Params params_;
  std::vector<DataType> inp_host_, zeros_input_host_, zeros_mean_var_host_,
      beta_host_, gamma_host_, mean_host_, var_host_, out_host_,
      workspace_host_;
  Provider provider_;
  Pointer inp_gpu_, beta_gpu_, gamma_gpu_, mean_gpu_, var_gpu_, out_gpu_;
  Pointer in_grad_gpu_, out_grad_gpu_, beta_grad_gpu_, gamma_grad_gpu_;
  Pointer running_mean_gpu_, running_var_gpu_, workspace_gpu_;

 public:
  SNNBatchnormExecutor(std::vector<DataType> const& inp_host,
                       std::vector<DataType> const& beta_host,
                       std::vector<DataType> const& gamma_host,
                       std::vector<DataType> const& mean_host,
                       std::vector<DataType> const& var_host,
                       Params const& params)
      : params_(params),
        inp_host_(inp_host),
        beta_host_(beta_host),
        gamma_host_(gamma_host),
        mean_host_(mean_host),
        var_host_(var_host),
        provider_{} {}

  void setup() {
    const auto input_size =
        params_.batch * params_.rows * params_.cols * params_.channels;

    out_host_.resize(input_size);
    this->inp_gpu_ = this->provider_.get_initialised_device_memory(
        this->inp_host_.size(), this->inp_host_);
    this->beta_gpu_ = this->provider_.get_initialised_device_memory(
        this->beta_host_.size(), this->beta_host_);
    this->gamma_gpu_ = this->provider_.get_initialised_device_memory(
        this->gamma_host_.size(), this->gamma_host_);
    this->out_gpu_ = this->provider_.get_initialised_device_memory(
        this->out_host_.size(), this->out_host_);

    // Forward inference
    // Inputs: input, gamma, beta, mean, var
    // Outputs: out
    if constexpr (IsForwardInference) {
      this->mean_gpu_ = this->provider_.get_initialised_device_memory(
          this->mean_host_.size(), this->mean_host_);
      this->var_gpu_ = this->provider_.get_initialised_device_memory(
          this->var_host_.size(), this->var_host_);
    }

    // Forward training
    // Inputs: input, gamma, beta
    // Outputs: running mean, running var, cache mean, cache var
    if constexpr (IsForwardTraining) {
      this->zeros_mean_var_host_.resize(this->params_.channels);
      this->mean_gpu_ = this->provider_.get_initialised_device_memory(
          this->zeros_mean_var_host_.size(), this->zeros_mean_var_host_);
      this->var_gpu_ = this->provider_.get_initialised_device_memory(
          this->zeros_mean_var_host_.size(), this->zeros_mean_var_host_);
      this->running_mean_gpu_ = this->provider_.get_initialised_device_memory(
          this->zeros_mean_var_host_.size(), this->zeros_mean_var_host_);
      this->running_var_gpu_ = this->provider_.get_initialised_device_memory(
          this->zeros_mean_var_host_.size(), this->zeros_mean_var_host_);
      sycldnn::batchnorm::compute_workspace_size<Direction>(params_);
      this->workspace_host_.resize(params_.required_workspace_size);
      this->workspace_gpu_ = this->provider_.get_initialised_device_memory(
          params_.required_workspace_size, this->workspace_host_);
    }

    // Backward
    // Inputs: out grad, input, gamma, cache mean, cache var
    // Outputs: in grad, gamma grad, beta grad
    if constexpr (IsGradient) {
      this->zeros_input_host_.resize(input_size);
      this->zeros_mean_var_host_.resize(this->params_.channels);
      this->mean_gpu_ = this->provider_.get_initialised_device_memory(
          this->mean_host_.size(), this->mean_host_);
      this->var_gpu_ = this->provider_.get_initialised_device_memory(
          this->var_host_.size(), this->var_host_);
      this->out_grad_gpu_ = this->provider_.get_initialised_device_memory(
          this->inp_host_.size(), this->inp_host_);
      this->in_grad_gpu_ = this->provider_.get_initialised_device_memory(
          this->zeros_input_host_.size(), this->zeros_input_host_);
      this->beta_grad_gpu_ = this->provider_.get_initialised_device_memory(
          this->zeros_mean_var_host_.size(), this->zeros_mean_var_host_);
      this->gamma_grad_gpu_ = this->provider_.get_initialised_device_memory(
          this->zeros_mean_var_host_.size(), this->zeros_mean_var_host_);
      sycldnn::batchnorm::compute_workspace_size<Direction>(params_);
      this->workspace_host_.resize(params_.required_workspace_size);
      this->workspace_gpu_ = this->provider_.get_initialised_device_memory(
          params_.required_workspace_size, this->workspace_host_);
    }
  }

  SNNStatus execute() {
    if constexpr (IsForwardInference) {
      return sycldnn::batchnorm::launch<DataType, Backend, Direction>(
          this->inp_gpu_, this->beta_gpu_, this->gamma_gpu_, this->mean_gpu_,
          this->var_gpu_, this->out_gpu_, this->params_,
          this->provider_.get_backend());
    } else if constexpr (IsForwardTraining) {  // Training
      return sycldnn::batchnorm::launch<DataType, Backend, Direction>(
          this->inp_gpu_, this->beta_gpu_, this->gamma_gpu_,
          this->running_mean_gpu_, this->running_var_gpu_, this->mean_gpu_,
          this->var_gpu_, this->workspace_gpu_, this->out_gpu_, this->params_,
          this->provider_.get_backend());
    } else {  // Gradient
      return sycldnn::batchnorm::launch<DataType, Backend, Direction>(
          this->inp_gpu_, this->out_grad_gpu_, this->in_grad_gpu_,
          this->gamma_gpu_, this->beta_grad_gpu_, this->gamma_grad_gpu_,
          this->mean_gpu_, this->var_gpu_, this->workspace_gpu_, this->params_,
          this->provider_.get_backend());
    }
  }

  void teardown() {
    if constexpr (IsForwardInference || IsForwardTraining) {
      this->provider_.deallocate_ptr(this->beta_gpu_);
    }
    if constexpr (IsForwardTraining) {
      this->provider_.deallocate_ptr(this->running_mean_gpu_);
      this->provider_.deallocate_ptr(this->running_var_gpu_);
    }
    if constexpr (IsGradient) {
      this->provider_.deallocate_ptr(this->out_grad_gpu_);
      this->provider_.deallocate_ptr(this->in_grad_gpu_);
      this->provider_.deallocate_ptr(this->beta_grad_gpu_);
      this->provider_.deallocate_ptr(this->gamma_grad_gpu_);
    }
    if constexpr (IsForwardTraining || IsGradient) {
      this->provider_.deallocate_ptr(this->workspace_gpu_);
    }
    this->provider_.deallocate_ptr(this->inp_gpu_);
    this->provider_.deallocate_ptr(this->gamma_gpu_);
    this->provider_.deallocate_ptr(this->mean_gpu_);
    this->provider_.deallocate_ptr(this->var_gpu_);
    this->provider_.deallocate_ptr(this->out_gpu_);
  }

  // Returns reference to backend from provider
  auto get_queue() { return this->provider_.get_backend().get_queue(); }

  // Retrieves the batchnorm result from device memory
  void copy_forward_output_to_host(std::vector<DataType>& host_data) {
    const auto output_size =
        params_.batch * params_.rows * params_.cols * params_.channels;
    this->provider_.copy_device_data_to_host(output_size, this->out_gpu_,
                                             host_data);
  }

  void copy_running_mean_var_to_host(std::vector<DataType>& running_mean_host,
                                     std::vector<DataType>& running_var_host) {
    const auto output_size = params_.channels;
    this->provider_.copy_device_data_to_host(
        output_size, this->running_mean_gpu_, running_mean_host);
    this->provider_.copy_device_data_to_host(
        output_size, this->running_var_gpu_, running_var_host);
  }

  void copy_gradient_outputs_to_host(std::vector<DataType>& in_grad_host,
                                     std::vector<DataType>& gamma_grad_host,
                                     std::vector<DataType>& beta_grad_host) {
    const auto output_size =
        params_.batch * params_.rows * params_.cols * params_.channels;
    this->provider_.copy_device_data_to_host(output_size, this->in_grad_gpu_,
                                             in_grad_host);
    this->provider_.copy_device_data_to_host(
        this->params_.channels, this->gamma_grad_gpu_, gamma_grad_host);
    this->provider_.copy_device_data_to_host(
        this->params_.channels, this->beta_grad_gpu_, beta_grad_host);
  }
};

}  // namespace bench
}  // namespace sycldnn

#endif  // SYCLDNN_BENCH_BATCHNORM_SNN_BATCHNORM_EXECUTOR_H
