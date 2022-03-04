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

#include "src/helpers/fast_div.h"
#include "src/helpers/tensor_index.h"

#include <stddef.h>
#include <cstdint>
#include <string>
#include <vector>

template <typename T>
struct TensorIndexTest : public ::testing::Test {
  void check_unflatten_2d(std::vector<T> const& indices, T const size,
                          std::vector<T> const& exp_0,
                          std::vector<T> const& exp_1) {
    SCOPED_TRACE("Not using fast integer divisions");
    check_unflatten_2d_impl<false>(indices, size, exp_0, exp_1);
  }
  void check_unflatten_2d_fast_div(std::vector<T> const& indices, T const size,
                                   std::vector<T> const& exp_0,
                                   std::vector<T> const& exp_1) {
    SCOPED_TRACE("Using fast integer divisions");
    check_unflatten_2d_impl<true>(indices, size, exp_0, exp_1);
  }
  void check_unflatten_3d(std::vector<T> const& indices, T const size1,
                          T const size2, std::vector<T> const& exp_0,
                          std::vector<T> const& exp_1,
                          std::vector<T> const& exp_2) {
    SCOPED_TRACE("Not using fast integer divisions");
    check_unflatten_3d_impl<false>(indices, size1, size2, exp_0, exp_1, exp_2);
  }
  void check_unflatten_3d_fast_div(std::vector<T> const& indices, T const size1,
                                   T const size2, std::vector<T> const& exp_0,
                                   std::vector<T> const& exp_1,
                                   std::vector<T> const& exp_2) {
    SCOPED_TRACE("Using fast integer divisions");
    check_unflatten_3d_impl<true>(indices, size1, size2, exp_0, exp_1, exp_2);
  }
  void check_unflatten_4d(std::vector<T> const& indices, T const size1,
                          T const size2, T const size3,
                          std::vector<T> const& exp_0,
                          std::vector<T> const& exp_1,
                          std::vector<T> const& exp_2,
                          std::vector<T> const& exp_3) {
    SCOPED_TRACE("Not using fast integer divisions");
    check_unflatten_4d_impl<false>(indices, size1, size2, size3, exp_0, exp_1,
                                   exp_2, exp_3);
  }
  void check_unflatten_4d_fast_div(std::vector<T> const& indices, T const size1,
                                   T const size2, T const size3,
                                   std::vector<T> const& exp_0,
                                   std::vector<T> const& exp_1,
                                   std::vector<T> const& exp_2,
                                   std::vector<T> const& exp_3) {
    SCOPED_TRACE("Using fast integer divisions");
    check_unflatten_4d_impl<true>(indices, size1, size2, size3, exp_0, exp_1,
                                  exp_2, exp_3);
  }

 private:
  template <bool UseFastDiv>
  void check_unflatten_2d_impl(std::vector<T> const& indices, T const size,
                               std::vector<T> const& exp_0,
                               std::vector<T> const& exp_1) {
    using DivType = typename sycldnn::fast_div::IndexDiv<T, UseFastDiv>::type;
    ASSERT_EQ(indices.size(), exp_0.size());
    ASSERT_EQ(indices.size(), exp_1.size());
    for (size_t i = 0; i < indices.size(); ++i) {
      SCOPED_TRACE("Index iteration: " + std::to_string(i));
      DivType fast_div{size};
      auto index =
          sycldnn::helpers::TensorIndexHelper<T, UseFastDiv>::unflatten2d(
              indices[i], fast_div, size);
      EXPECT_EQ(exp_0[i], index.s0);
      EXPECT_EQ(exp_1[i], index.s1);
    }
  }
  template <bool UseFastDiv>
  void check_unflatten_3d_impl(std::vector<T> const& indices, T const size1,
                               T const size2, std::vector<T> const& exp_0,
                               std::vector<T> const& exp_1,
                               std::vector<T> const& exp_2) {
    using DivType = typename sycldnn::fast_div::IndexDiv<T, UseFastDiv>::type;
    ASSERT_EQ(indices.size(), exp_0.size());
    ASSERT_EQ(indices.size(), exp_1.size());
    ASSERT_EQ(indices.size(), exp_2.size());
    for (size_t i = 0; i < indices.size(); ++i) {
      SCOPED_TRACE("Index iteration: " + std::to_string(i));
      DivType fast_div1{size1};
      DivType fast_div2{size2};
      auto index =
          sycldnn::helpers::TensorIndexHelper<T, UseFastDiv>::unflatten3d(
              indices[i], fast_div1, size1, fast_div2, size2);
      EXPECT_EQ(exp_0[i], index.s0);
      EXPECT_EQ(exp_1[i], index.s1);
      EXPECT_EQ(exp_2[i], index.s2);
    }
  }
  template <bool UseFastDiv>
  void check_unflatten_4d_impl(std::vector<T> const& indices, T const size1,
                               T const size2, T const size3,
                               std::vector<T> const& exp_0,
                               std::vector<T> const& exp_1,
                               std::vector<T> const& exp_2,
                               std::vector<T> const& exp_3) {
    using DivType = typename sycldnn::fast_div::IndexDiv<T, UseFastDiv>::type;
    ASSERT_EQ(indices.size(), exp_0.size());
    ASSERT_EQ(indices.size(), exp_1.size());
    ASSERT_EQ(indices.size(), exp_2.size());
    ASSERT_EQ(indices.size(), exp_3.size());
    for (size_t i = 0; i < indices.size(); ++i) {
      SCOPED_TRACE("Index iteration: " + std::to_string(i));
      DivType fast_div1{size1};
      DivType fast_div2{size2};
      DivType fast_div3{size3};
      auto index =
          sycldnn::helpers::TensorIndexHelper<T, UseFastDiv>::unflatten4d(
              indices[i], fast_div1, size1, fast_div2, size2, fast_div3, size3);
      EXPECT_EQ(exp_0[i], index.s0);
      EXPECT_EQ(exp_1[i], index.s1);
      EXPECT_EQ(exp_2[i], index.s2);
      EXPECT_EQ(exp_3[i], index.s3);
    }
  }
};
using IntegerTypes = ::testing::Types<int32_t, int64_t>;
TYPED_TEST_SUITE(TensorIndexTest, IntegerTypes);

TYPED_TEST(TensorIndexTest, Unflatten2DSize4) {
  std::vector<TypeParam> indices = {0, 1, 2,  3,  4,  5,  6,  7,
                                    8, 9, 10, 11, 12, 13, 14, 15};
  TypeParam size = 4;
  std::vector<TypeParam> exp_0 = {0, 0, 0, 0, 1, 1, 1, 1,
                                  2, 2, 2, 2, 3, 3, 3, 3};
  std::vector<TypeParam> exp_1 = {0, 1, 2, 3, 0, 1, 2, 3,
                                  0, 1, 2, 3, 0, 1, 2, 3};
  this->check_unflatten_2d(indices, size, exp_0, exp_1);
  this->check_unflatten_2d_fast_div(indices, size, exp_0, exp_1);
}
TYPED_TEST(TensorIndexTest, Unflatten2DSize7) {
  std::vector<TypeParam> indices = {0, 1, 2,  3,  4,  5,  6,  7,
                                    8, 9, 10, 11, 12, 13, 14, 15};
  TypeParam size = 7;
  std::vector<TypeParam> exp_0 = {0, 0, 0, 0, 0, 0, 0, 1,
                                  1, 1, 1, 1, 1, 1, 2, 2};
  std::vector<TypeParam> exp_1 = {0, 1, 2, 3, 4, 5, 6, 0,
                                  1, 2, 3, 4, 5, 6, 0, 1};
  this->check_unflatten_2d(indices, size, exp_0, exp_1);
  this->check_unflatten_2d_fast_div(indices, size, exp_0, exp_1);
}
TYPED_TEST(TensorIndexTest, Unflatten3DSize1x3) {
  std::vector<TypeParam> indices = {0, 1, 2,  3,  4,  5,  6,  7,
                                    8, 9, 10, 11, 12, 13, 14, 15};
  TypeParam size1 = 1;
  TypeParam size2 = 3;
  std::vector<TypeParam> exp_0 = {0, 0, 0, 1, 1, 1, 2, 2,
                                  2, 3, 3, 3, 4, 4, 4, 5};
  std::vector<TypeParam> exp_1 = {0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<TypeParam> exp_2 = {0, 1, 2, 0, 1, 2, 0, 1,
                                  2, 0, 1, 2, 0, 1, 2, 0};
  this->check_unflatten_3d(indices, size1, size2, exp_0, exp_1, exp_2);
  // Cannot use fast divisions when one of the divisors is 1
}
TYPED_TEST(TensorIndexTest, Unflatten3DSize3x1) {
  std::vector<TypeParam> indices = {0, 1, 2,  3,  4,  5,  6,  7,
                                    8, 9, 10, 11, 12, 13, 14, 15};
  TypeParam size1 = 3;
  TypeParam size2 = 1;
  std::vector<TypeParam> exp_0 = {0, 0, 0, 1, 1, 1, 2, 2,
                                  2, 3, 3, 3, 4, 4, 4, 5};
  std::vector<TypeParam> exp_1 = {0, 1, 2, 0, 1, 2, 0, 1,
                                  2, 0, 1, 2, 0, 1, 2, 0};
  std::vector<TypeParam> exp_2 = {0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0};
  this->check_unflatten_3d(indices, size1, size2, exp_0, exp_1, exp_2);
  // Cannot use fast divisions when one of the divisors is 1
}
TYPED_TEST(TensorIndexTest, Unflatten3DSize3x3) {
  std::vector<TypeParam> indices = {0, 1, 2,  3,  4,  5,  6,  7,
                                    8, 9, 10, 11, 12, 13, 14, 15};
  TypeParam size1 = 3;
  TypeParam size2 = 3;
  std::vector<TypeParam> exp_0 = {0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 1, 1, 1, 1, 1, 1, 1};
  std::vector<TypeParam> exp_1 = {0, 0, 0, 1, 1, 1, 2, 2,
                                  2, 0, 0, 0, 1, 1, 1, 2};
  std::vector<TypeParam> exp_2 = {0, 1, 2, 0, 1, 2, 0, 1,
                                  2, 0, 1, 2, 0, 1, 2, 0};
  this->check_unflatten_3d(indices, size1, size2, exp_0, exp_1, exp_2);
  this->check_unflatten_3d_fast_div(indices, size1, size2, exp_0, exp_1, exp_2);
}
TYPED_TEST(TensorIndexTest, Unflatten3DSize3x7) {
  std::vector<TypeParam> indices = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                                    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
  TypeParam size1 = 3;
  TypeParam size2 = 7;
  std::vector<TypeParam> exp_0 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<TypeParam> exp_1 = {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2,
                                  2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1};
  std::vector<TypeParam> exp_2 = {0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0,
                                  1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1};
  this->check_unflatten_3d(indices, size1, size2, exp_0, exp_1, exp_2);
  this->check_unflatten_3d_fast_div(indices, size1, size2, exp_0, exp_1, exp_2);
}
TYPED_TEST(TensorIndexTest, Unflatten3DSize7x2) {
  std::vector<TypeParam> indices = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                                    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
  TypeParam size1 = 7;
  TypeParam size2 = 3;
  std::vector<TypeParam> exp_0 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<TypeParam> exp_1 = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4,
                                  5, 5, 5, 6, 6, 6, 0, 0, 0, 1, 1, 1, 2, 2, 2};
  std::vector<TypeParam> exp_2 = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
                                  0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
  this->check_unflatten_3d(indices, size1, size2, exp_0, exp_1, exp_2);
  this->check_unflatten_3d_fast_div(indices, size1, size2, exp_0, exp_1, exp_2);
}
TYPED_TEST(TensorIndexTest, Unflatten4DSize3x3x3) {
  std::vector<TypeParam> indices = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                                    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
  TypeParam size1 = 3;
  TypeParam size2 = 3;
  TypeParam size3 = 3;
  std::vector<TypeParam> exp_0 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1};
  std::vector<TypeParam> exp_1 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                                  1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0};
  std::vector<TypeParam> exp_2 = {0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1,
                                  2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0};
  std::vector<TypeParam> exp_3 = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
                                  0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2};
  this->check_unflatten_4d(indices, size1, size2, size3, exp_0, exp_1, exp_2,
                           exp_3);
  this->check_unflatten_4d_fast_div(indices, size1, size2, size3, exp_0, exp_1,
                                    exp_2, exp_3);
}
TYPED_TEST(TensorIndexTest, Unflatten4DSize2x2x5) {
  std::vector<TypeParam> indices = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                                    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
  TypeParam size1 = 2;
  TypeParam size2 = 2;
  TypeParam size3 = 5;
  std::vector<TypeParam> exp_0 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<TypeParam> exp_1 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                                  1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<TypeParam> exp_2 = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                                  1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  std::vector<TypeParam> exp_3 = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
                                  0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4};
  this->check_unflatten_4d(indices, size1, size2, size3, exp_0, exp_1, exp_2,
                           exp_3);
  this->check_unflatten_4d_fast_div(indices, size1, size2, size3, exp_0, exp_1,
                                    exp_2, exp_3);
}
TYPED_TEST(TensorIndexTest, Unflatten4DSize2x5x2) {
  std::vector<TypeParam> indices = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                                    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
  TypeParam size1 = 2;
  TypeParam size2 = 5;
  TypeParam size3 = 2;
  std::vector<TypeParam> exp_0 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<TypeParam> exp_1 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                                  1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<TypeParam> exp_2 = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 1, 1, 2,
                                  2, 3, 3, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4};
  std::vector<TypeParam> exp_3 = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                                  1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  this->check_unflatten_4d(indices, size1, size2, size3, exp_0, exp_1, exp_2,
                           exp_3);
  this->check_unflatten_4d_fast_div(indices, size1, size2, size3, exp_0, exp_1,
                                    exp_2, exp_3);
}
TYPED_TEST(TensorIndexTest, Unflatten4DSize5x2x2) {
  std::vector<TypeParam> indices = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                                    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                    20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
  TypeParam size1 = 5;
  TypeParam size2 = 2;
  TypeParam size3 = 2;
  std::vector<TypeParam> exp_0 = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<TypeParam> exp_1 = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3,
                                  3, 4, 4, 4, 4, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2};
  std::vector<TypeParam> exp_2 = {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1,
                                  1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0};
  std::vector<TypeParam> exp_3 = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                                  1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  this->check_unflatten_4d(indices, size1, size2, size3, exp_0, exp_1, exp_2,
                           exp_3);
  this->check_unflatten_4d_fast_div(indices, size1, size2, size3, exp_0, exp_1,
                                    exp_2, exp_3);
}
