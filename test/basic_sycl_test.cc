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
#include <gtest/gtest.h>

#include <array>

#include <CL/sycl.hpp>

class HostSet;
class DeviceSet;

TEST(BasicSycl, construct_queue_with_selector) {
  cl::sycl::default_selector selector;
  cl::sycl::queue queue{selector};
}
#ifndef SYCL_IMPLEMENTATION_ONEAPI  // No host device in DPC++
TEST(BasicSycl, host_set_float) {
  static constexpr auto write_mode = cl::sycl::access::mode::write;
  constexpr size_t num_elems = 10;
  cl::sycl::host_selector selector;
  cl::sycl::queue queue{selector};
  std::array<float, num_elems> base_mem{};
  {
    cl::sycl::buffer<float, 1> bufA{base_mem.data(),
                                    cl::sycl::range<1>{num_elems}};
    queue.submit([&](cl::sycl::handler& cgh) {
      auto accessorA = bufA.get_access<write_mode>(cgh);

      cgh.parallel_for<HostSet>(cl::sycl::range<1>{num_elems},
                                [=](cl::sycl::item<1> item) {
                                  const auto id = item.get_id(0);
                                  accessorA[id] = id * 0.1f;
                                });
    });
  }
  for (size_t i = 0; i < num_elems; ++i) {
    ASSERT_FLOAT_EQ(static_cast<float>(i) * 0.1f, base_mem[i]);
  }
}
#endif
TEST(BasicSycl, device_set_float) {
  static constexpr auto write_mode = cl::sycl::access::mode::write;
  constexpr size_t num_elems = 10;
  cl::sycl::default_selector selector;
  cl::sycl::queue queue{selector};
  std::array<float, num_elems> base_mem{};
  {
    cl::sycl::buffer<float, 1> bufA{base_mem.data(),
                                    cl::sycl::range<1>{num_elems}};
    queue.submit([&](cl::sycl::handler& cgh) {
      auto accessorA = bufA.get_access<write_mode>(cgh);

      cgh.parallel_for<DeviceSet>(cl::sycl::range<1>{num_elems},
                                  [=](cl::sycl::item<1> item) {
                                    const auto id = item.get_id(0);
                                    accessorA[id] = id * 0.1f;
                                  });
    });
  }
  for (size_t i = 0; i < num_elems; ++i) {
    ASSERT_FLOAT_EQ(static_cast<float>(i) * 0.1f, base_mem[i]);
  }
}
