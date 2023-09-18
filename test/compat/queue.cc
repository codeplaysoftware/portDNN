
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

#include "portdnn/compat/utils.hpp"

using namespace sycldnn;
using namespace sycldnn::compat;

TEST(Handle, create) {
  SNNHandle handle;
  const auto status = SNNCreate(handle);

  EXPECT_TRUE(status == StatusCode::OK);
}

TEST(Handle, invalid_properties) {
  SNNHandle handle;
  const auto status = SNNCreate(handle, sycl::default_selector(),
                                {sycl::property::queue::enable_profiling()});

  EXPECT_TRUE(status == StatusCode::InvalidParameter);
}

TEST(StreamTest, basic) {
  sycl::queue q1{sycl::default_selector(), sycl::property::queue::in_order()};

  SNNHandle handle;
  SNNCreate(handle);
  const auto status = queueSet(handle, q1);

  EXPECT_TRUE(status == StatusCode::OK);
  EXPECT_TRUE(handle.getQueue() == q1);
}

TEST(StreamTest, mem_access) {
  auto fill_ptr = [&](sycl::queue queue, int* data, int fill_val,
                      size_t num_elems) {
    return queue.parallel_for(sycl::range(num_elems), [=](sycl::item<1> item) {
      data[item.get_id()] = fill_val;
    });
  };

  constexpr size_t num_elems = 10;
  constexpr int fill_val = 1, fill_val2 = 2;
  SNNHandle handle;
  SNNCreate(handle);

  auto* data =
      sycl::malloc<int>(num_elems, handle.getQueue(), sycl::usm::alloc::shared);
  auto ev = fill_ptr(handle.getQueue(), data, fill_val, num_elems);
  ev.wait_and_throw();

  for (size_t i = 0; i < num_elems; ++i) {
    EXPECT_TRUE(data[i] == fill_val);
  }

  sycl::queue q1{sycl::default_selector(), sycl::property::queue::in_order()};
  const auto status = queueSet(handle, q1);

  EXPECT_TRUE(status == StatusCode::OK);
  EXPECT_TRUE(handle.getQueue() == q1);

  ev = fill_ptr(handle.getQueue(), data, fill_val2, num_elems);
  ev.wait_and_throw();

  for (size_t i = 0; i < num_elems; ++i) {
    EXPECT_TRUE(data[i] == fill_val2);
  }

  EXPECT_TRUE(status == StatusCode::OK);
  EXPECT_TRUE(handle.getQueue() == q1);
}

TEST(StreamTest, different_devices) {
  SNNHandle handle;
  sycl::queue q;
  try {
    SNNCreate(handle, sycl::cpu_selector());
    q = sycl::queue{sycl::gpu_selector()};
  } catch (const sycl::exception&) {
    GTEST_SKIP() << "Test skipped as it requires both a CPU and a GPU device";
  }
  const auto status = queueSet(handle, q);

  EXPECT_TRUE(status == StatusCode::InvalidParameter);
}

TEST(StreamTest, in_order_queue) {
  SNNHandle handle;
  SNNCreate(handle);
  sycl::queue q{sycl::default_selector()};
  const auto status = queueSet(handle, q);

  EXPECT_TRUE(status == StatusCode::InvalidParameter);
}
