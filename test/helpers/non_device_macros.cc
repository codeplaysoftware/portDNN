/*
 * Copyright 2018 Codeplay Software Ltd
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
#include <gtest/gtest.h>

#include "sycldnn/helpers/macros.h"

TEST(NonDeviceHelperMacro, NoAlwaysInlineOnHost) {
  auto test_str = SNN_MAKE_STRING(SNN_ALWAYS_INLINE);
  auto expected = "";
  ASSERT_STREQ(expected, test_str);
}
TEST(NonDeviceHelperMacro, NoPragmaUnrollOnHost) {
  auto test_str = SNN_MAKE_STRING(SNN_PRAGMA_UNROLL);
  auto expected = "";
  ASSERT_STREQ(expected, test_str);
}
