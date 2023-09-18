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
#ifndef PORTDNN_TEST_REDUCE_REDUCE_FIXTURE_H_
#define PORTDNN_TEST_REDUCE_REDUCE_FIXTURE_H_

#include <gtest/gtest.h>
#include <vector>

#include "portdnn/backend/snn_usm_backend.h"

#include "portdnn/helpers/scope_exit.h"
#include "portdnn/reduce/launch.h"
#include "test/backend/backend_test_fixture.h"
#include "test/gen/iota_initialised_data.h"
#include "test/helpers/dependency_check.h"
#include "test/helpers/float_comparison.h"

template <typename T, typename Op>
struct ReduceEventFixture
    : public BackendTestFixture<sycldnn::backend::SNNUSMBackend> {
  using DataType = T;

 protected:
  void run(int batches, int outer, int inner, DataType max_val) {
    size_t input_size = batches * outer * inner;
    size_t output_size = batches * inner;

    std::vector<DataType> input_data =
        iota_initialised_data(input_size, max_val);
    std::vector<DataType> output_data =
        iota_initialised_data(output_size, max_val);

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    {
      auto input_gpu =
          provider.get_initialised_device_memory(input_size, input_data);
      auto output_gpu =
          provider.get_initialised_device_memory(output_size, output_data);
      SNN_ON_SCOPE_EXIT {
        provider.deallocate_ptr(input_gpu);
        provider.deallocate_ptr(output_gpu);
      };

      dependency_test_params dep_test_params;
      cl::sycl::event dependee_e = create_event(backend, dep_test_params);

      auto status = sycldnn::reduce::launch<DataType, Op>(
          input_gpu, output_gpu, batches, outer, inner, backend,
          std::vector<cl::sycl::event>{dependee_e});

      check_dependency(dependee_e, status.event, backend, dep_test_params);
    }
  }
};

#endif  // PORTDNN_TEST_REDUCE_REDUCE_FIXTURE_H_
