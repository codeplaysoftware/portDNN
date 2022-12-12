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

#ifndef SYCLDNN_TEST_HELPERS_DEPENDENCY_CHECK_H_
#define SYCLDNN_TEST_HELPERS_DEPENDENCY_CHECK_H_

#include <CL/sycl.hpp>

struct dependency_test_params {
  uint64_t* event_mem_h = nullptr;
  uint64_t* event_mem_d = nullptr;
};

template <typename USMbackend>
cl::sycl::event create_event(USMbackend backend, dependency_test_params& params,
                             uint64_t cpy_size = 10000) {
  params.event_mem_h =
      cl::sycl::malloc_host<uint64_t>(cpy_size, backend.get_queue());
  params.event_mem_d =
      cl::sycl::malloc_device<uint64_t>(cpy_size, backend.get_queue());
  return backend.get_queue().memcpy(params.event_mem_d, params.event_mem_h,
                                    cpy_size * sizeof(uint64_t));
}

template <typename USMbackend>
void check_dependency(cl::sycl::event dependee_e, cl::sycl::event depender_e,
                      USMbackend backend, dependency_test_params& params) {
  cl::sycl::info::event_command_status depender_status =
      depender_e.get_info<cl::sycl::info::event::command_execution_status>();

  while (depender_status != cl::sycl::info::event_command_status::running &&
         depender_status != cl::sycl::info::event_command_status::complete) {
    depender_status =
        depender_e.get_info<cl::sycl::info::event::command_execution_status>();
  }

  EXPECT_EQ(
      dependee_e.get_info<cl::sycl::info::event::command_execution_status>(),
      cl::sycl::info::event_command_status::complete);

  cl::sycl::free(params.event_mem_h, backend.get_queue());
  cl::sycl::free(params.event_mem_d, backend.get_queue());

  params.event_mem_h = nullptr;
  params.event_mem_d = nullptr;
}

#endif  // SYCLDNN_TEST_HELPERS_DEPENDENCY_CHECK_H_
