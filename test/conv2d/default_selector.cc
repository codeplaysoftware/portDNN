/*
 * Copyright 2018 Codeplay Software Ltd.
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

#include "sycldnn/conv2d/selector/default_selector.h"
#include "sycldnn/conv2d/selector/selector.h"

TEST(DefaultSelectorTest, GetReturnsNonNull) {
  // Construct a SYCL queue with the default selector.
  cl::sycl::queue q;

  // Use the queue to get a device reference, then validate that we return a
  // non-null selector.
  auto selector = sycldnn::conv2d::get_default_selector(q.get_device());
  EXPECT_TRUE(nullptr != selector);
}
