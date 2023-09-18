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
#ifndef PORTDNN_TEST_BINARYOP_FIXTURE_H_
#define PORTDNN_TEST_BINARYOP_FIXTURE_H_

#include <gtest/gtest.h>
#include <vector>

#include "portdnn/binaryop/launch.h"
#include "portdnn/helpers/dims.h"
#include "portdnn/helpers/scope_exit.h"
#include "test/backend/backend_test_fixture.h"
#include "test/gen/iota_initialised_data.h"
#include "test/helpers/dependency_check.h"
#include "test/helpers/float_comparison.h"

template <typename T, typename Op>
struct BinaryOpEventFixture
    : public BackendTestFixture<sycldnn::backend::SNNUSMBackend> {
  using DataType = T;

 protected:
  void run(const unsigned int exp_size, sycldnn::binaryop::BinaryParams params,
           DataType max_val) {
    std::vector<DataType> lhs_data = iota_initialised_data(
        sycldnn::helpers::get_total_size(params.lhs_dims), max_val);
    std::vector<DataType> rhs_data = iota_initialised_data(
        sycldnn::helpers::get_total_size(params.rhs_dims), max_val);
    std::vector<DataType> out_data = iota_initialised_data(exp_size, max_val);

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    {
      auto lhs_gpu =
          provider.get_initialised_device_memory(lhs_data.size(), lhs_data);
      auto rhs_gpu =
          provider.get_initialised_device_memory(rhs_data.size(), rhs_data);
      auto out_gpu =
          provider.get_initialised_device_memory(out_data.size(), out_data);
      SNN_ON_SCOPE_EXIT {
        provider.deallocate_ptr(lhs_gpu);
        provider.deallocate_ptr(rhs_gpu);
        provider.deallocate_ptr(out_gpu);
      };

      dependency_test_params dep_test_params;
      cl::sycl::event dependee_e = create_event(backend, dep_test_params);

      auto status = sycldnn::binaryop::launch<DataType, Op>(
          lhs_gpu, rhs_gpu, out_gpu, params, backend,
          std::vector<cl::sycl::event>{dependee_e});

      ASSERT_EQ(sycldnn::StatusCode::OK, status.status);

      check_dependency(dependee_e, status.event, backend, dep_test_params);
    }
  }
};

#endif  // PORTDNN_TEST_BINARYOP_FIXTURE_H_
