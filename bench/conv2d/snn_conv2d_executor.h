/*
 * Copyright Codeplay Software Ltd.
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
#ifndef SYCLDNN_BENCH_SNN_CONV2D_EXECUTOR_H_
#define SYCLDNN_BENCH_SNN_CONV2D_EXECUTOR_H_

#include "sycldnn/conv2d/conv_type.h"
#include "sycldnn/conv2d/launch.h"
#include "sycldnn/conv2d/params.h"
#include "sycldnn/conv2d/selector/selector.h"
#include "sycldnn/conv2d/sizes.h"
#include "sycldnn/conv2d/workspace_size.h"

#include "sycldnn/transpose/launch.h"

#include "src/backend/backend_provider.h"

#include "bench/common/event_handling.h"
#include "bench/common/verification_utils.h"

namespace sycldnn {
namespace bench {

/**
 * Get the required size for the workspace buffer.
 *
 * Query the required workspace size and the available allocation size on the
 * device, then choose a size based on this. The size is chosen to be as large
 * as it can be while still fitting in memory. If the smallest size is still
 * too large to be allocated then 0 is returned and fall back to using
 * separate temporary buffers.
 */
template <typename ConvType>
size_t compute_workspace_size(conv2d::Conv2DParams const& params,
                              cl::sycl::device device,
                              conv2d::Selector& selector) {
  auto workspace_size_struct =
      sycldnn::conv2d::query_workspace_size<ConvType>(params, selector);
  // Query the max allocation size on the device. As the input, output and
  // filter tensors also need to be allocated we conservatively divide the
  // alloc size by 4.
  auto max_alloc_size =
      device.get_info<cl::sycl::info::device::max_mem_alloc_size>() / 4;

  if (workspace_size_struct.recommended_size < max_alloc_size) {
    return workspace_size_struct.recommended_size;
  } else {
    if (workspace_size_struct.required_size < max_alloc_size) {
      // If the recommended size is too large but the required size is small
      // enough to be allocated then choose a size in between.
      auto n_multiples = max_alloc_size / workspace_size_struct.required_size;
      return workspace_size_struct.required_size * n_multiples;
    } else {
      // If even the required size is too large then we cannot use a
      // workspace.
      return 0u;
    }
  }
}

template <typename DataType, typename ConvType, typename Selector,
          typename Backend, bool InOrderQueue = false,
          bool UseTranspose = false>
class SNNConv2DExecutor {
  using Conv2DParams = conv2d::Conv2DParams;
  using Conv2DSizes = conv2d::ConvSizes;
  using Provider = typename backend::BackendProvider<Backend, InOrderQueue>;
  using Pointer = typename Provider::template Pointer<DataType>;

  Pointer inp_gpu_;
  Pointer fil_gpu_;
  Pointer out_gpu_;
  Pointer ws_gpu_;
  Pointer inp_t_gpu_;
  Pointer fil_t_gpu_;
  Pointer out_t_gpu_;

  Conv2DParams params_;
  Conv2DSizes sizes_;

  Selector& selector_;
  Provider provider_;

  std::vector<DataType> inp_host_;
  std::vector<DataType> fil_host_;

  int ws_size_;
  bool deallocate_ws_ = false;

 public:
  SNNConv2DExecutor(std::vector<DataType> const& inp_host,
                    std::vector<DataType> const& fil_host,
                    Conv2DParams const& params, Selector& selector)
      : params_{params},
        sizes_{conv2d::get_sizes<ConvType>(params)},
        selector_{selector},
        provider_{},
        inp_host_(inp_host.begin(), inp_host.end()),
        fil_host_(fil_host.begin(), fil_host.end()) {}

  // Allocate and initialise device memory
  void setup() {
    auto& backend = provider_.get_backend();

    std::vector<DataType> out_host(this->sizes_.output_size,
                                   static_cast<DataType>(0));
    // Initialise device memory and copy host inputs to initialised memory
    this->inp_gpu_ = this->provider_.get_initialised_device_memory(
        this->sizes_.input_size, this->inp_host_);
    this->fil_gpu_ = this->provider_.get_initialised_device_memory(
        this->sizes_.filter_size, this->fil_host_);
    this->out_gpu_ = this->provider_.get_initialised_device_memory(
        this->sizes_.output_size, out_host);

    if constexpr (UseTranspose) {
      this->params_.input_format = DataFormat::NHWC;
      this->params_.filter_format = FilterFormat::HWCF;
      this->inp_t_gpu_ = this->provider_.get_initialised_device_memory(
          this->sizes_.input_size, this->inp_host_);
      this->fil_t_gpu_ = this->provider_.get_initialised_device_memory(
          this->sizes_.filter_size, this->fil_host_);
      this->out_t_gpu_ = this->provider_.get_initialised_device_memory(
          this->sizes_.output_size, out_host);
    }

    // Initialise workspace if needed
    this->ws_size_ = compute_workspace_size<ConvType>(
        this->params_, backend.get_queue().get_device(), this->selector_);
    if (this->ws_size_ > 0) {
      std::vector<DataType> workspace_vals(this->ws_size_,
                                           static_cast<DataType>(0));
      this->ws_gpu_ = this->provider_.get_initialised_device_memory(
          this->ws_size_, workspace_vals);
      this->deallocate_ws_ = true;
    }
  }

  // Execute the SNN convolution once.
  SNNStatus execute() {
    // Maybe transpose input data NCHW->NHWC
    auto trans_inp_status = this->maybe_transpose_input();
    if (trans_inp_status.status != StatusCode::OK) return trans_inp_status;
    // Maybe transpose filter data FCHW->HWCF
    auto trans_fil_status = this->maybe_transpose_filter();
    if (trans_fil_status.status != StatusCode::OK) return trans_fil_status;

    std::vector<cl::sycl::event> deps = {trans_inp_status.event,
                                         trans_fil_status.event};

    auto& backend = this->provider_.get_backend();
    SNNStatus conv_status;
    if constexpr (sycldnn::backend::is_usm_backend_v<Backend>) {
      conv_status = conv2d::launch<DataType, ConvType>(
          this->inp_gpu_, this->fil_gpu_, this->out_gpu_, this->params_,
          this->selector_, backend, this->ws_gpu_, this->ws_size_, deps);
    } else {
      conv_status = conv2d::launch<DataType, ConvType>(
          this->inp_gpu_, this->fil_gpu_, this->out_gpu_, this->params_,
          this->selector_, backend, this->ws_gpu_, this->ws_size_);
    }

    // Maybe transpose output data NFHW->NHWF
    if (conv_status.status != StatusCode::OK) return conv_status;
    auto out_trans_status = this->maybe_transpose_output(conv_status.event);
    return out_trans_status;
  }

  // Deallocates device memory
  void teardown() {
    this->provider_.deallocate_ptr(this->out_gpu_);
    this->provider_.deallocate_ptr(this->fil_gpu_);
    this->provider_.deallocate_ptr(this->inp_gpu_);
    if (this->deallocate_ws_) {
      this->provider_.deallocate_ptr(this->ws_gpu_);
    }
  }

  SNNStatus maybe_transpose_input() {
    if constexpr (UseTranspose) {
      const auto spatial_sizes =
          sycldnn::conv2d::get_spatial_sizes<ConvType>(params_);
      const auto channel_sizes =
          sycldnn::conv2d::get_channel_sizes<ConvType>(params_);
      std::vector<int> const dimensions = {
          params_.batch, static_cast<int>(channel_sizes.input_size),
          static_cast<int>(spatial_sizes.input_size)};
      std::vector<int> const permutation = {0, 2, 1};

      return transpose::launch<DataType>(inp_t_gpu_, inp_gpu_, dimensions,
                                         permutation,
                                         this->provider_.get_backend());
    } else {
      return {{}, StatusCode::OK};
    }
  }

  SNNStatus maybe_transpose_filter() {
    if constexpr (UseTranspose) {
      std::vector<int> dimensions;
      std::vector<int> permutation;
      auto const spatial_sizes =
          sycldnn::conv2d::get_spatial_sizes<ConvType>(params_);
      if constexpr (std::is_same_v<
                        ConvType, sycldnn::conv2d::conv_type::FilterBackprop>) {
        auto const channel_sizes =
            sycldnn::conv2d::get_channel_sizes<ConvType>(params_);
        dimensions = {params_.batch,
                      static_cast<int>(channel_sizes.filter_size),
                      static_cast<int>(spatial_sizes.filter_size)};
        permutation = {0, 2, 1};
      } else {
        dimensions = {params_.features, params_.channels,
                      static_cast<int>(spatial_sizes.filter_size)};
        permutation = {2, 1, 0};
      }
      return transpose::launch<DataType>(fil_t_gpu_, fil_gpu_, dimensions,
                                         permutation,
                                         this->provider_.get_backend());
    } else {
      return {{}, StatusCode::OK};
    }
  }

  SNNStatus maybe_transpose_output(cl::sycl::event const& event) {
    if constexpr (UseTranspose) {
      std::vector<int> dimensions;
      std::vector<int> permutation;
      auto const spatial_sizes =
          sycldnn::conv2d::get_spatial_sizes<ConvType>(params_);
      if constexpr (std::is_same_v<
                        ConvType, sycldnn::conv2d::conv_type::FilterBackprop>) {
        dimensions = {static_cast<int>(spatial_sizes.output_size),
                      params_.channels, params_.features};
        permutation = {2, 1, 0};
      } else {
        auto const channel_sizes =
            sycldnn::conv2d::get_channel_sizes<ConvType>(params_);
        dimensions = {params_.batch,
                      static_cast<int>(spatial_sizes.output_size),
                      static_cast<int>(channel_sizes.output_size)};
        permutation = {0, 2, 1};
      }

      if constexpr (sycldnn::backend::is_usm_backend_v<Backend>) {
        return transpose::launch<DataType>(
            out_gpu_, out_t_gpu_, dimensions, permutation,
            this->provider_.get_backend(), {event});
      } else {
        return transpose::launch<DataType>(out_gpu_, out_t_gpu_, dimensions,
                                           permutation,
                                           this->provider_.get_backend());
      }
    } else {
      return {event, StatusCode::OK};
    }
  }
  // Returns reference to backend from provider
  auto get_queue() { return this->provider_.get_backend().get_queue(); }

  // Retrieves the convolution result from device memory
  void copy_output_to_host(std::vector<DataType>& host_data) {
    if constexpr (UseTranspose) {
      this->provider_.copy_device_data_to_host(this->sizes_.output_size,
                                               this->out_t_gpu_, host_data);
    } else {
      this->provider_.copy_device_data_to_host(this->sizes_.output_size,
                                               this->out_gpu_, host_data);
    }
  }
};
}  // namespace bench
}  // namespace sycldnn
#endif  // SYCLDNN_BENCH_SNN_CONV2D_EXECUTOR_H_
