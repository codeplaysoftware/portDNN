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

#include "src/helpers/vector_element.h"

#include <CL/sycl.hpp>

namespace {
// Helper functions to compare expected values to actual values in SYCL vectors.
// The explicit cast from a swizzled vector is required to ensure GTest uses
// the correct data type and not the implementation defined swizzle type.
// The numerous overloads of this function are required as there is no
// programmatic way to access vector elements, and we don't want to rely on
// helpers::vector_element in the test for that function.
template <typename DType>
void check_vector_matches(std::array<DType, 1> exp,
                          cl::sycl::vec<DType, 1> const& vec) {
  EXPECT_EQ(exp[0], DType{vec.s0()});
}
template <typename DType>
void check_vector_matches(std::array<DType, 2> exp,
                          cl::sycl::vec<DType, 2> const& vec) {
  EXPECT_EQ(exp[0], DType{vec.s0()});
  EXPECT_EQ(exp[1], DType{vec.s1()});
}
template <typename DType>
void check_vector_matches(std::array<DType, 3> exp,
                          cl::sycl::vec<DType, 3> const& vec) {
  EXPECT_EQ(exp[0], DType{vec.s0()});
  EXPECT_EQ(exp[1], DType{vec.s1()});
  EXPECT_EQ(exp[2], DType{vec.s2()});
}
template <typename DType>
void check_vector_matches(std::array<DType, 4> exp,
                          cl::sycl::vec<DType, 4> const& vec) {
  EXPECT_EQ(exp[0], DType{vec.s0()});
  EXPECT_EQ(exp[1], DType{vec.s1()});
  EXPECT_EQ(exp[2], DType{vec.s2()});
  EXPECT_EQ(exp[3], DType{vec.s3()});
}
template <typename DType>
void check_vector_matches(std::array<DType, 8> exp,
                          cl::sycl::vec<DType, 8> const& vec) {
  EXPECT_EQ(exp[0], DType{vec.s0()});
  EXPECT_EQ(exp[1], DType{vec.s1()});
  EXPECT_EQ(exp[2], DType{vec.s2()});
  EXPECT_EQ(exp[3], DType{vec.s3()});
  EXPECT_EQ(exp[4], DType{vec.s4()});
  EXPECT_EQ(exp[5], DType{vec.s5()});
  EXPECT_EQ(exp[6], DType{vec.s6()});
  EXPECT_EQ(exp[7], DType{vec.s7()});
}
template <typename DType>
void check_vector_matches(std::array<DType, 16> exp,
                          cl::sycl::vec<DType, 16> const& vec) {
  EXPECT_EQ(exp[0], DType{vec.s0()});
  EXPECT_EQ(exp[1], DType{vec.s1()});
  EXPECT_EQ(exp[2], DType{vec.s2()});
  EXPECT_EQ(exp[3], DType{vec.s3()});
  EXPECT_EQ(exp[4], DType{vec.s4()});
  EXPECT_EQ(exp[5], DType{vec.s5()});
  EXPECT_EQ(exp[6], DType{vec.s6()});
  EXPECT_EQ(exp[7], DType{vec.s7()});
  EXPECT_EQ(exp[8], DType{vec.s8()});
  EXPECT_EQ(exp[9], DType{vec.s9()});
  EXPECT_EQ(exp[10], DType{vec.sA()});
  EXPECT_EQ(exp[11], DType{vec.sB()});
  EXPECT_EQ(exp[12], DType{vec.sC()});
  EXPECT_EQ(exp[13], DType{vec.sD()});
  EXPECT_EQ(exp[14], DType{vec.sE()});
  EXPECT_EQ(exp[15], DType{vec.sF()});
}
}  // namespace

template <typename T>
struct VectorElementTest : public ::testing::Test {};

using NumericTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(VectorElementTest, NumericTypes);

TYPED_TEST(VectorElementTest, NonVectorTypes) {
  TypeParam a = 0;
  TypeParam val = sycldnn::helpers::vector_element::get(a, 0);
  EXPECT_EQ(a, val);

  TypeParam const b = 10;
  sycldnn::helpers::vector_element::set(a, 0, b);
  EXPECT_EQ(b, a);
}
TYPED_TEST(VectorElementTest, Vector1DType) {
  using Vec = cl::sycl::vec<TypeParam, 1>;
  TypeParam a = 0;
  Vec vec{a};
  TypeParam val = sycldnn::helpers::vector_element::get(vec, 0);
  EXPECT_EQ(a, val);

  TypeParam const b = 10;
  sycldnn::helpers::vector_element::set(vec, 0, b);
  check_vector_matches({b}, vec);
}
TYPED_TEST(VectorElementTest, Vector2DType) {
  using Vec = cl::sycl::vec<TypeParam, 2>;
  TypeParam a0 = 0;
  TypeParam a1 = 1;
  Vec vec{a0, a1};
  TypeParam val_0 = sycldnn::helpers::vector_element::get(vec, 0);
  EXPECT_EQ(a0, val_0);

  TypeParam val_1 = sycldnn::helpers::vector_element::get(vec, 1);
  EXPECT_EQ(a1, val_1);

  TypeParam const b0 = 10;
  sycldnn::helpers::vector_element::set(vec, 0, b0);
  check_vector_matches({b0, a1}, vec);

  TypeParam const b1 = 15;
  sycldnn::helpers::vector_element::set(vec, 1, b1);
  check_vector_matches({b0, b1}, vec);
}
TYPED_TEST(VectorElementTest, Vector3DType) {
  using Vec = cl::sycl::vec<TypeParam, 3>;
  TypeParam a0 = 0;
  TypeParam a1 = 1;
  TypeParam a2 = 2;
  Vec vec{a0, a1, a2};
  TypeParam val_0 = sycldnn::helpers::vector_element::get(vec, 0);
  EXPECT_EQ(a0, val_0);

  TypeParam val_1 = sycldnn::helpers::vector_element::get(vec, 1);
  EXPECT_EQ(a1, val_1);

  TypeParam val_2 = sycldnn::helpers::vector_element::get(vec, 2);
  EXPECT_EQ(a2, val_2);

  TypeParam const b0 = 10;
  sycldnn::helpers::vector_element::set(vec, 0, b0);
  check_vector_matches({b0, a1, a2}, vec);

  TypeParam const b1 = 15;
  sycldnn::helpers::vector_element::set(vec, 1, b1);
  check_vector_matches({b0, b1, a2}, vec);

  TypeParam const b2 = 20;
  sycldnn::helpers::vector_element::set(vec, 2, b2);
  check_vector_matches({b0, b1, b2}, vec);
}
TYPED_TEST(VectorElementTest, Vector4DType) {
  using Vec = cl::sycl::vec<TypeParam, 4>;
  TypeParam a0 = 0;
  TypeParam a1 = 1;
  TypeParam a2 = 2;
  TypeParam a3 = 3;
  Vec vec{a0, a1, a2, a3};
  TypeParam val_0 = sycldnn::helpers::vector_element::get(vec, 0);
  EXPECT_EQ(a0, val_0);

  TypeParam val_1 = sycldnn::helpers::vector_element::get(vec, 1);
  EXPECT_EQ(a1, val_1);

  TypeParam val_2 = sycldnn::helpers::vector_element::get(vec, 2);
  EXPECT_EQ(a2, val_2);

  TypeParam val_3 = sycldnn::helpers::vector_element::get(vec, 3);
  EXPECT_EQ(a3, val_3);

  TypeParam const b0 = 10;
  sycldnn::helpers::vector_element::set(vec, 0, b0);
  check_vector_matches({b0, a1, a2, a3}, vec);

  TypeParam const b1 = 15;
  sycldnn::helpers::vector_element::set(vec, 1, b1);
  check_vector_matches({b0, b1, a2, a3}, vec);

  TypeParam const b2 = 20;
  sycldnn::helpers::vector_element::set(vec, 2, b2);
  check_vector_matches({b0, b1, b2, a3}, vec);

  TypeParam const b3 = 30;
  sycldnn::helpers::vector_element::set(vec, 3, b3);
  check_vector_matches({b0, b1, b2, b3}, vec);
}
TYPED_TEST(VectorElementTest, Vector8DType) {
  using Vec = cl::sycl::vec<TypeParam, 8>;
  TypeParam a0 = 0;
  TypeParam a1 = 1;
  TypeParam a2 = 2;
  TypeParam a3 = 3;
  TypeParam a4 = 4;
  TypeParam a5 = 5;
  TypeParam a6 = 6;
  TypeParam a7 = 7;
  Vec vec{a0, a1, a2, a3, a4, a5, a6, a7};
  TypeParam val_0 = sycldnn::helpers::vector_element::get(vec, 0);
  EXPECT_EQ(a0, val_0);

  TypeParam val_1 = sycldnn::helpers::vector_element::get(vec, 1);
  EXPECT_EQ(a1, val_1);

  TypeParam val_2 = sycldnn::helpers::vector_element::get(vec, 2);
  EXPECT_EQ(a2, val_2);

  TypeParam val_3 = sycldnn::helpers::vector_element::get(vec, 3);
  EXPECT_EQ(a3, val_3);

  TypeParam val_4 = sycldnn::helpers::vector_element::get(vec, 4);
  EXPECT_EQ(a4, val_4);

  TypeParam val_5 = sycldnn::helpers::vector_element::get(vec, 5);
  EXPECT_EQ(a5, val_5);

  TypeParam val_6 = sycldnn::helpers::vector_element::get(vec, 6);
  EXPECT_EQ(a6, val_6);

  TypeParam val_7 = sycldnn::helpers::vector_element::get(vec, 7);
  EXPECT_EQ(a7, val_7);

  TypeParam const b0 = 10;
  sycldnn::helpers::vector_element::set(vec, 0, b0);
  check_vector_matches({b0, a1, a2, a3, a4, a5, a6, a7}, vec);

  TypeParam const b1 = 15;
  sycldnn::helpers::vector_element::set(vec, 1, b1);
  check_vector_matches({b0, b1, a2, a3, a4, a5, a6, a7}, vec);

  TypeParam const b2 = 20;
  sycldnn::helpers::vector_element::set(vec, 2, b2);
  check_vector_matches({b0, b1, b2, a3, a4, a5, a6, a7}, vec);

  TypeParam const b3 = 30;
  sycldnn::helpers::vector_element::set(vec, 3, b3);
  check_vector_matches({b0, b1, b2, b3, a4, a5, a6, a7}, vec);

  TypeParam const b4 = 40;
  sycldnn::helpers::vector_element::set(vec, 4, b4);
  check_vector_matches({b0, b1, b2, b3, b4, a5, a6, a7}, vec);

  TypeParam const b5 = 50;
  sycldnn::helpers::vector_element::set(vec, 5, b5);
  check_vector_matches({b0, b1, b2, b3, b4, b5, a6, a7}, vec);

  TypeParam const b6 = 60;
  sycldnn::helpers::vector_element::set(vec, 6, b6);
  check_vector_matches({b0, b1, b2, b3, b4, b5, b6, a7}, vec);

  TypeParam const b7 = 70;
  sycldnn::helpers::vector_element::set(vec, 7, b7);
  check_vector_matches({b0, b1, b2, b3, b4, b5, b6, b7}, vec);
}
// NOLINTNEXTLINE(google-readability-function-size)
TYPED_TEST(VectorElementTest, Vector16DType) {
  using Vec = cl::sycl::vec<TypeParam, 16>;
  TypeParam a0 = 0;
  TypeParam a1 = 1;
  TypeParam a2 = 2;
  TypeParam a3 = 3;
  TypeParam a4 = 4;
  TypeParam a5 = 5;
  TypeParam a6 = 6;
  TypeParam a7 = 7;
  TypeParam a8 = 8;
  TypeParam a9 = 9;
  TypeParam a10 = 10;
  TypeParam a11 = 11;
  TypeParam a12 = 12;
  TypeParam a13 = 13;
  TypeParam a14 = 14;
  TypeParam a15 = 15;
  Vec vec{a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15};
  TypeParam val_0 = sycldnn::helpers::vector_element::get(vec, 0);
  EXPECT_EQ(a0, val_0);

  TypeParam val_1 = sycldnn::helpers::vector_element::get(vec, 1);
  EXPECT_EQ(a1, val_1);

  TypeParam val_2 = sycldnn::helpers::vector_element::get(vec, 2);
  EXPECT_EQ(a2, val_2);

  TypeParam val_3 = sycldnn::helpers::vector_element::get(vec, 3);
  EXPECT_EQ(a3, val_3);

  TypeParam val_4 = sycldnn::helpers::vector_element::get(vec, 4);
  EXPECT_EQ(a4, val_4);

  TypeParam val_5 = sycldnn::helpers::vector_element::get(vec, 5);
  EXPECT_EQ(a5, val_5);

  TypeParam val_6 = sycldnn::helpers::vector_element::get(vec, 6);
  EXPECT_EQ(a6, val_6);

  TypeParam val_7 = sycldnn::helpers::vector_element::get(vec, 7);
  EXPECT_EQ(a7, val_7);

  TypeParam val_8 = sycldnn::helpers::vector_element::get(vec, 8);
  EXPECT_EQ(a8, val_8);

  TypeParam val_9 = sycldnn::helpers::vector_element::get(vec, 9);
  EXPECT_EQ(a9, val_9);

  TypeParam val_10 = sycldnn::helpers::vector_element::get(vec, 10);
  EXPECT_EQ(a10, val_10);

  TypeParam val_11 = sycldnn::helpers::vector_element::get(vec, 11);
  EXPECT_EQ(a11, val_11);

  TypeParam val_12 = sycldnn::helpers::vector_element::get(vec, 12);
  EXPECT_EQ(a12, val_12);

  TypeParam val_13 = sycldnn::helpers::vector_element::get(vec, 13);
  EXPECT_EQ(a13, val_13);

  TypeParam val_14 = sycldnn::helpers::vector_element::get(vec, 14);
  EXPECT_EQ(a14, val_14);

  TypeParam val_15 = sycldnn::helpers::vector_element::get(vec, 15);
  EXPECT_EQ(a15, val_15);

  TypeParam const b0 = 10;
  sycldnn::helpers::vector_element::set(vec, 0, b0);
  check_vector_matches(
      {b0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15},
      vec);

  TypeParam const b1 = 15;
  sycldnn::helpers::vector_element::set(vec, 1, b1);
  check_vector_matches(
      {b0, b1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15},
      vec);

  TypeParam const b2 = 20;
  sycldnn::helpers::vector_element::set(vec, 2, b2);
  check_vector_matches(
      {b0, b1, b2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15},
      vec);

  TypeParam const b3 = 30;
  sycldnn::helpers::vector_element::set(vec, 3, b3);
  check_vector_matches(
      {b0, b1, b2, b3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15},
      vec);

  TypeParam const b4 = 40;
  sycldnn::helpers::vector_element::set(vec, 4, b4);
  check_vector_matches(
      {b0, b1, b2, b3, b4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15},
      vec);

  TypeParam const b5 = 50;
  sycldnn::helpers::vector_element::set(vec, 5, b5);
  check_vector_matches(
      {b0, b1, b2, b3, b4, b5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15},
      vec);

  TypeParam const b6 = 60;
  sycldnn::helpers::vector_element::set(vec, 6, b6);
  check_vector_matches(
      {b0, b1, b2, b3, b4, b5, b6, a7, a8, a9, a10, a11, a12, a13, a14, a15},
      vec);

  TypeParam const b7 = 70;
  sycldnn::helpers::vector_element::set(vec, 7, b7);
  check_vector_matches(
      {b0, b1, b2, b3, b4, b5, b6, b7, a8, a9, a10, a11, a12, a13, a14, a15},
      vec);

  TypeParam const b8 = 80;
  sycldnn::helpers::vector_element::set(vec, 8, b8);
  check_vector_matches(
      {b0, b1, b2, b3, b4, b5, b6, b7, b8, a9, a10, a11, a12, a13, a14, a15},
      vec);

  TypeParam const b9 = 90;
  sycldnn::helpers::vector_element::set(vec, 9, b9);
  check_vector_matches(
      {b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, a10, a11, a12, a13, a14, a15},
      vec);

  TypeParam const b10 = 100;
  sycldnn::helpers::vector_element::set(vec, 10, b10);
  check_vector_matches(
      {b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, a11, a12, a13, a14, a15},
      vec);

  TypeParam const b11 = 110;
  sycldnn::helpers::vector_element::set(vec, 11, b11);
  check_vector_matches(
      {b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, a12, a13, a14, a15},
      vec);

  TypeParam const b12 = 120;
  sycldnn::helpers::vector_element::set(vec, 12, b12);
  check_vector_matches(
      {b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, a13, a14, a15},
      vec);

  TypeParam const b13 = 130;
  sycldnn::helpers::vector_element::set(vec, 13, b13);
  check_vector_matches(
      {b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, a14, a15},
      vec);

  TypeParam const b14 = 140;
  sycldnn::helpers::vector_element::set(vec, 14, b14);
  check_vector_matches(
      {b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, a15},
      vec);

  TypeParam const b15 = 150;
  sycldnn::helpers::vector_element::set(vec, 15, b15);
  check_vector_matches(
      {b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15},
      vec);
}
