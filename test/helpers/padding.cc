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

#include "portdnn/helpers/padding.h"
#include "portdnn/padding_mode.h"

#include <stddef.h>
#include <cstdint>
#include <string>
#include <vector>

template <typename Index>
struct PaddingTest : public ::testing::Test {
 protected:
  void test_values(std::vector<Index> const& inputs, Index window, Index stride,
                   sycldnn::PaddingMode type,
                   std::vector<Index> const& expected_padding,
                   std::vector<Index> const& expected_output) {
    ASSERT_EQ(expected_padding.size(), inputs.size());
    ASSERT_EQ(expected_output.size(), inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      test_single_value(inputs[i], window, stride, type, expected_padding[i],
                        expected_output[i]);
    }
  }
  void test_single_value(Index input, Index window, Index stride,
                         sycldnn::PaddingMode type, Index expected_padding,
                         Index expected_output) {
    auto padding =
        sycldnn::helpers::calculate_padding(input, window, stride, type);
    EXPECT_EQ(expected_padding, padding.padding);
    EXPECT_EQ(expected_output, padding.output);
  }
};

using IndexTypes = ::testing::Types<int32_t, int64_t>;
TYPED_TEST_SUITE(PaddingTest, IndexTypes);

TYPED_TEST(PaddingTest, ValidWindow1Stride1) {
  TypeParam window = 1;
  TypeParam stride = 1;
  auto type = sycldnn::PaddingMode::VALID;
  std::vector<TypeParam> inputs = {3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_pad = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<TypeParam> exp_out = {3, 4, 5, 6, 7, 8, 9, 10};
  this->test_values(inputs, window, stride, type, exp_pad, exp_out);
}
TYPED_TEST(PaddingTest, ValidWindow1Stride2) {
  TypeParam window = 1;
  TypeParam stride = 2;
  auto type = sycldnn::PaddingMode::VALID;
  std::vector<TypeParam> inputs = {3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_pad = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<TypeParam> exp_out = {2, 2, 3, 3, 4, 4, 5, 5};
  this->test_values(inputs, window, stride, type, exp_pad, exp_out);
}
TYPED_TEST(PaddingTest, SameWindow1Stride1) {
  TypeParam window = 1;
  TypeParam stride = 1;
  auto type = sycldnn::PaddingMode::SAME;
  std::vector<TypeParam> inputs = {3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_pad = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<TypeParam> exp_out = {3, 4, 5, 6, 7, 8, 9, 10};
  this->test_values(inputs, window, stride, type, exp_pad, exp_out);
}
TYPED_TEST(PaddingTest, SameWindow1Stride2) {
  TypeParam window = 1;
  TypeParam stride = 2;
  auto type = sycldnn::PaddingMode::SAME;
  std::vector<TypeParam> inputs = {3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_pad = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<TypeParam> exp_out = {2, 2, 3, 3, 4, 4, 5, 5};
  this->test_values(inputs, window, stride, type, exp_pad, exp_out);
}
TYPED_TEST(PaddingTest, ValidWindow3Stride1) {
  TypeParam window = 3;
  TypeParam stride = 1;
  auto type = sycldnn::PaddingMode::VALID;
  std::vector<TypeParam> inputs = {3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_pad = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<TypeParam> exp_out = {1, 2, 3, 4, 5, 6, 7, 8};
  this->test_values(inputs, window, stride, type, exp_pad, exp_out);
}
TYPED_TEST(PaddingTest, ValidWindow3Stride2) {
  TypeParam window = 3;
  TypeParam stride = 2;
  auto type = sycldnn::PaddingMode::VALID;
  std::vector<TypeParam> inputs = {3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_pad = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<TypeParam> exp_out = {1, 1, 2, 2, 3, 3, 4, 4};
  this->test_values(inputs, window, stride, type, exp_pad, exp_out);
}
TYPED_TEST(PaddingTest, SameWindow3Stride1) {
  TypeParam window = 3;
  TypeParam stride = 1;
  auto type = sycldnn::PaddingMode::SAME;
  std::vector<TypeParam> inputs = {3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_pad = {1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<TypeParam> exp_out = {3, 4, 5, 6, 7, 8, 9, 10};
  this->test_values(inputs, window, stride, type, exp_pad, exp_out);
}
TYPED_TEST(PaddingTest, SameWindow3Stride2) {
  TypeParam window = 3;
  TypeParam stride = 2;
  auto type = sycldnn::PaddingMode::SAME;
  std::vector<TypeParam> inputs = {3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_pad = {1, 0, 1, 0, 1, 0, 1, 0};
  std::vector<TypeParam> exp_out = {2, 2, 3, 3, 4, 4, 5, 5};
  this->test_values(inputs, window, stride, type, exp_pad, exp_out);
}
TYPED_TEST(PaddingTest, SameWindow3Stride3) {
  TypeParam window = 3;
  TypeParam stride = 3;
  auto type = sycldnn::PaddingMode::SAME;
  std::vector<TypeParam> inputs = {3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<TypeParam> exp_pad = {0, 1, 0, 0, 1, 0, 0, 1};
  std::vector<TypeParam> exp_out = {1, 2, 2, 2, 3, 3, 3, 4};
  this->test_values(inputs, window, stride, type, exp_pad, exp_out);
}
