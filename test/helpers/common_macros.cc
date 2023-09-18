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
#include <gtest/gtest.h>

#include "portdnn/helpers/macros.h"

TEST(HelperMacro, BasicStringify) {
  auto test_str = SNN_MAKE_STRING(abc);
  auto expected = "abc";
  ASSERT_STREQ(expected, test_str);
}
TEST(HelperMacro, MacroStringify) {
#define XYZ xyz
  auto test_str = SNN_MAKE_STRING(XYZ);
  auto expected = "xyz";
  ASSERT_STREQ(expected, test_str);
#undef XYZ
}
