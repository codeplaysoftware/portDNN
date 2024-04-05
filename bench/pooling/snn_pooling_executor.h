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
#ifndef SYCLDNN_BENCH_SNN_POOLING_EXECUTOR_H_
#define SYCLDNN_BENCH_SNN_POOLING_EXECUTOR_H_

#include "sycldnn/pooling/launch.h"

#include "src/backend/backend_provider.h"

#include "bench/common/event_handling.h"
#include "bench/common/verification_utils.h"

namespace sycldnn {
namespace bench {

template <typename DataType, template <typename> class Operator,
          typename Direction, typename Backend, bool InOrderQueue>
class SNNPoolingExecutor {
  using PoolingParams = pooling::PoolingParams;
  using PoolingSizes = pooling::PoolingSizes;
  using Provider = typename backend::BackendProvider<Backend, InOrderQueue>;
  using Pointer = typename Provider::template Pointer<DataType>;

  Pointer inp_gpu_;
  Pointer out_gpu_;

  PoolingParams params_;
  PoolingSizes sizes_;

  Provider provider_;

  std::vector<DataType> inp_host_;

 public:
  SNNPoolingExecutor(std::vector<DataType> const& inp_host,
                     PoolingParams const& params)
      : params_{params},
        sizes_{pooling::get_sizes<Direction>(params)},
        provider_{},
        inp_host_(inp_host.begin(), inp_host.end()) {}

  // Allocate and initialise device memory
  void setup() {
    std::vector<DataType> out_host(this->sizes_.output_size,
                                   static_cast<DataType>(0));
    // Initialise device memory and copy host inputs to initialised memory
    this->inp_gpu_ = this->provider_.get_initialised_device_memory(
        this->sizes_.input_size, this->inp_host_);
    this->out_gpu_ = this->provider_.get_initialised_device_memory(
        this->sizes_.output_size, out_host);
  }

  // Execute the SNN pooling once.
  SNNStatus execute() {
    auto& backend = this->provider_.get_backend();
    return sycldnn::pooling::launch<DataType, Operator, Direction>(
        this->inp_gpu_, this->out_gpu_, this->params_, backend);
  }

  // Deallocates device memory
  void teardown() {
    this->provider_.deallocate_ptr(this->out_gpu_);
    this->provider_.deallocate_ptr(this->inp_gpu_);
  }

  // Returns cl::sycl::queue
  auto get_queue() { return this->provider_.get_backend.get_queue(); }

  // Retrieves the pooling result from device memory
  void copy_output_to_host(std::vector<DataType>& host_data) {
    this->provider_.copy_device_data_to_host(this->sizes_.output_size,
                                             this->out_gpu_, host_data);
  }
};

template <typename DataType, typename Backend, bool InOrderQueue>
class SNNPoolingExecutor<DataType, pooling::Max, pooling::Backpropagate,
                         Backend, InOrderQueue> {
  using PoolingParams = pooling::PoolingParams;
  using PoolingSizes = pooling::PoolingSizes;
  using Provider = typename backend::BackendProvider<Backend>;
  using Pointer = typename Provider::template Pointer<DataType>;

  Pointer inp_gpu_;
  Pointer inp_grad_gpu_;
  Pointer out_gpu_;
  Pointer out_grad_gpu_;

  PoolingParams params_;
  PoolingSizes fwd_sizes_;
  PoolingSizes back_sizes_;

  Provider provider_;

  std::vector<DataType> inp_host_;
  std::vector<DataType> out_grad_host_;

 public:
  SNNPoolingExecutor(std::vector<DataType> const& inp_host,
                     std::vector<DataType> const& out_grad_host,
                     PoolingParams const& params)
      : params_{params},
        fwd_sizes_{pooling::get_sizes<pooling::Forward>(params)},
        back_sizes_{pooling::get_sizes<pooling::Backpropagate>(params)},
        provider_{},
        inp_host_(inp_host),
        out_grad_host_(out_grad_host) {}

  // Allocate and initialise device memory
  void setup() {
    std::vector<DataType> inp_grad_host(this->back_sizes_.output_size,
                                        static_cast<DataType>(0));
    // Initialise device memory and copy host inputs to initialised memory
    this->inp_gpu_ = this->provider_.get_initialised_device_memory(
        this->fwd_sizes_.input_size, this->inp_host_);
    this->out_grad_gpu_ = this->provider_.get_initialised_device_memory(
        this->back_sizes_.input_size, this->out_grad_host_);
    this->out_gpu_ = this->provider_.get_initialised_device_memory(
        this->fwd_sizes_.output_size, this->out_grad_host_);
    this->inp_grad_gpu_ = this->provider_.get_initialised_device_memory(
        this->back_sizes_.output_size, inp_grad_host);
  }

  // Execute the SNN pooling once.
  SNNStatus execute() {
    auto& backend = this->provider_.get_backend();
    return sycldnn::pooling::launch<DataType, pooling::Max,
                                    pooling::Backpropagate>(
        this->inp_gpu_, this->out_gpu_, this->out_grad_gpu_,
        this->inp_grad_gpu_, this->params_, backend);
  }

  // Deallocates device memory
  void teardown() {
    this->provider_.deallocate_ptr(this->inp_gpu_);
    this->provider_.deallocate_ptr(this->inp_grad_gpu_);
    this->provider_.deallocate_ptr(this->out_gpu_);
    this->provider_.deallocate_ptr(this->out_grad_gpu_);
  }
  // Retrieves the pooling result from device memory
  void copy_output_to_host(std::vector<DataType>& host_data) {
    this->provider_.copy_device_data_to_host(this->back_sizes_.output_size,
                                             this->inp_grad_gpu_, host_data);
  }
};

}  // namespace bench
}  // namespace sycldnn
#endif  // SYCLDNN_BENCH_SNN_POOLING_EXECUTOR_H_
