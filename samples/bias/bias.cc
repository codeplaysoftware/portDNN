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

#include "portdnn/backend/snn_backend.h"

#include "portdnn/binaryop/launch.h"
#include "portdnn/binaryop/operators.h"
#include "portdnn/binaryop/params.h"

#include "portdnn/status.h"

#include <iostream>

#include <CL/sycl.hpp>

namespace snn = sycldnn;
#ifndef SYCL_IMPLEMENTATION_ONEAPI
namespace sycl = cl::sycl;
#endif

using Backend = snn::backend::SNNBackend;
using DeviceMem = Backend::pointer_type<float>;

int main() {
  sycl::queue q([](sycl::exception_list l) {
    for (auto e : l) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception& e) {
        std::cout << e.what() << " " << e.get_cl_code() << "\n";
      }
    }
  });
  Backend backend(q);
  snn::binaryop::BinaryParams params{};
  params.lhs_dims = {256, 16};
  params.rhs_dims = {1, 16};
  auto lhs_size = sycldnn::helpers::get_total_size(params.lhs_dims);
  auto rhs_size = sycldnn::helpers::get_total_size(params.rhs_dims);

  std::vector<float> in(lhs_size, 10.0);
  std::vector<float> bias(rhs_size, 0.5);
  std::vector<float> out(lhs_size, 0.0);

  auto input = backend.allocate<float>(in.size());
  auto biases = backend.allocate<float>(bias.size());
  auto output = backend.allocate<float>(out.size());
  auto buf_in = input.get_buffer();
  auto buf_bias = biases.get_buffer();
  auto event = q.submit([&](sycl::handler& cgh) {
    auto acc_in = buf_in.get_access<sycl::access::mode::write>(cgh);
    cgh.copy(in.data(), acc_in);
  });
  event.wait_and_throw();

  event = q.submit([&](sycl::handler& cgh) {
    auto acc_bias = buf_bias.get_access<sycl::access::mode::write>(cgh);
    cgh.copy(bias.data(), acc_bias);
  });
  event.wait_and_throw();

  auto st = std::chrono::high_resolution_clock::now();
  auto bias_event = snn::binaryop::launch<float, snn::binaryop::Add>(
      input, biases, output, params, backend);
  bias_event.event.wait_and_throw();
  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "Finished Execution of the Bias-Add event after time "
            << (end - st).count() << "ns\n\n";
  return 0;
}
