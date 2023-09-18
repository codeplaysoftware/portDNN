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

// Manually define __SYCL_DEVICE_ONLY__ to simulate compiling on device
#define __SYCL_DEVICE_ONLY__
#include "portdnn/helpers/macros.h"

TEST(DeviceHelperMacro, AlwaysInlineOnDevice) {
// These tests depend on certain preprocessor behaviour which is not
// compatible with MSVC, so avoid compiling these tests with the MSVC
// compiler.
#ifndef SNN_WINDOWS
  auto test_str = SNN_MAKE_STRING(SNN_ALWAYS_INLINE);
#if SNN_HAS_ATTRIBUTE(always_inline)
  auto expected = "__attribute__((always_inline))";
#else
  auto expected = "";
#endif
  ASSERT_STREQ(expected, test_str);
#endif
}
TEST(DeviceHelperMacro, PragmaUnrollOnDevice) {
#ifndef SNN_WINDOWS
  auto test_str = SNN_MAKE_STRING(SNN_PRAGMA_UNROLL);
#ifdef __clang__
  auto expected = "_Pragma(\"clang loop unroll(enable) interleave(enable)\")";
#else
  auto expected = "";
#endif
  ASSERT_STREQ(expected, test_str);
#endif
}
