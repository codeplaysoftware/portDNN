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

#include "tools/layer.h"

#include <CL/sycl.hpp>

namespace sycldnn {
template <typename DType, typename Backend>
class Network {
  using DeviceMem = typename Backend::template pointer_type<DType>;
  std::vector<std::unique_ptr<Layer<DType, Backend>>> network_;
  std::vector<DType>& output_;
  Backend& backend_;

 public:
  Network(Backend& backend, std::vector<DType>& output)
      : network_{}, output_{output}, backend_{backend} {}

  // Layers are their own types, number of parameters differs between each
  void add_layer(Layer<DType, Backend>* layer) { network_.emplace_back(layer); }

  // Runs each layer, checks for exceptions after every layer
  sycldnn::SNNStatus test() {
    sycldnn::SNNStatus status;
    for (auto& layer : network_) {
      status = layer->run();
      status.event.wait_and_throw();
    }
    return dump_network_output();
  }

  sycldnn::SNNStatus run() {
    sycldnn::SNNStatus status;
    for (auto& layer : network_) {
      status = layer->run();
    }
    return status;
  }

  DeviceMem get_output() { return network_.back()->get_output(); }

  DeviceMem get_output(int layer_number) {
    return network_[layer_number]->get_output();
  }

  size_t get_network_size() const { return network_.size(); }

  size_t get_output_size() const { return network_.back()->get_output_size(); }

  sycldnn::SNNStatus dump_network_output() {
    DeviceMem out = this->get_output();
    auto count = this->get_output_size();
    output_.resize(count);

    auto buf_out = out.get_buffer();
    auto event = backend_.get_queue().submit([&](cl::sycl::handler& cgh) {
      auto acc_out =
          buf_out.template get_access<cl::sycl::access::mode::read>(cgh);

      cgh.copy(acc_out, output_.data());
    });
    return {event, sycldnn::StatusCode::OK};
  }
};
}  // namespace sycldnn
