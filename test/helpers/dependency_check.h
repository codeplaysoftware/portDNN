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

#ifndef PORTDNN_TEST_HELPERS_DEPENDENCY_CHECK_H_
#define PORTDNN_TEST_HELPERS_DEPENDENCY_CHECK_H_

#include <CL/sycl.hpp>
#include "portdnn/helpers/macros.h"

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
void check_dependency(cl::sycl::event e1, cl::sycl::event e2,
                      USMbackend backend, dependency_test_params& params) {
  // This test is not as thorough as it ought to be, due to a likely long-term
  // bug with DPC++'s PI plugin: https://github.com/intel/llvm/issues/8132
  // Basically for (at least) CUDA and Level Zero, kernels are reported
  // 'running' as soon as they have been submitted, *not* when they actually
  // start running. This means that all we can safely test are:
  // Test 1. Once e2 is "running", make sure e1 is "running" or complete
  // Test 2. Once e2 is complete, ensure e1 is also complete
  cl::sycl::info::event_command_status e2_status =
      e2.get_info<cl::sycl::info::event::command_execution_status>();
  while (e2_status != cl::sycl::info::event_command_status::running &&
         e2_status != cl::sycl::info::event_command_status::complete) {
    e2_status = e2.get_info<cl::sycl::info::event::command_execution_status>();
  }

  // Test 1.
  cl::sycl::info::event_command_status e1_status =
      e1.get_info<cl::sycl::info::event::command_execution_status>();
  bool running_or_done =
      e1_status == cl::sycl::info::event_command_status::running ||
      e1_status == cl::sycl::info::event_command_status::complete;
  EXPECT_TRUE(running_or_done);

  // Test 2.
  while (e2_status != cl::sycl::info::event_command_status::complete) {
    e2_status = e2.get_info<cl::sycl::info::event::command_execution_status>();
  }
  e1_status = e1.get_info<cl::sycl::info::event::command_execution_status>();
  bool done = e1_status == cl::sycl::info::event_command_status::complete;
  EXPECT_TRUE(done);

  backend.get_queue().wait_and_throw();

  cl::sycl::free(params.event_mem_h, backend.get_queue());
  cl::sycl::free(params.event_mem_d, backend.get_queue());

  params.event_mem_h = nullptr;
  params.event_mem_d = nullptr;
}

#endif  // PORTDNN_TEST_HELPERS_DEPENDENCY_CHECK_H_
