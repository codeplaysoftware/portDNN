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

#include "src/helpers/math.h"
#include "src/helpers/vector_element.h"

#include <stddef.h>
#include <string>
#include <vector>

#include <CL/sycl.hpp>

template <typename T>
struct MathTest : public ::testing::Test {
  template <typename TestType>
  void check_mad_values(std::vector<T> const& a, std::vector<T> const& b,
                        std::vector<T> const& c, std::vector<T> const& exp) {
    ASSERT_EQ(exp.size(), a.size());
    ASSERT_EQ(exp.size(), b.size());
    ASSERT_EQ(exp.size(), c.size());
    for (size_t i = 0; i < exp.size(); ++i) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      auto val = sycldnn::helpers::math::mad(TestType{a[i]}, TestType{b[i]},
                                             TestType{c[i]});
      all_equal(exp[i], val);
    }
  }

 private:
  template <int N>
  void all_equal(T exp, cl::sycl::vec<T, N> const& val) {
    for (int i = 0; i < N; ++i) {
      SCOPED_TRACE("Vector element: " + std::to_string(i));
      EXPECT_EQ(exp, sycldnn::helpers::vector_element::get(val, i));
    }
  }
  void all_equal(T exp, T val) { EXPECT_EQ(exp, val); }
};
using NumericTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(MathTest, NumericTypes);

TYPED_TEST(MathTest, NonVectorValues) {
  using TestType = TypeParam;
  std::vector<TypeParam> a = {1, 1, 1, 1, 1, 1, 1, 1, 1,
                              2, 2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<TypeParam> b = {1, 1, 1, 2, 2, 2, 3, 3, 3,
                              1, 1, 1, 2, 2, 2, 3, 3, 3};
  std::vector<TypeParam> c = {1, 2, 3, 1, 2, 3, 1, 2, 3,
                              1, 2, 3, 1, 2, 3, 1, 2, 3};
  std::vector<TypeParam> exp = {2, 3, 4, 3, 4, 5, 4, 5, 6,
                                3, 4, 5, 5, 6, 7, 7, 8, 9};
  this->template check_mad_values<TestType>(a, b, c, exp);
}
TYPED_TEST(MathTest, Vector1DValues) {
  using TestType = cl::sycl::vec<TypeParam, 1>;
  std::vector<TypeParam> a = {1, 1, 1, 1, 1, 1, 1, 1, 1,
                              2, 2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<TypeParam> b = {1, 1, 1, 2, 2, 2, 3, 3, 3,
                              1, 1, 1, 2, 2, 2, 3, 3, 3};
  std::vector<TypeParam> c = {1, 2, 3, 1, 2, 3, 1, 2, 3,
                              1, 2, 3, 1, 2, 3, 1, 2, 3};
  std::vector<TypeParam> exp = {2, 3, 4, 3, 4, 5, 4, 5, 6,
                                3, 4, 5, 5, 6, 7, 7, 8, 9};
  this->template check_mad_values<TestType>(a, b, c, exp);
}
TYPED_TEST(MathTest, Vector2DValues) {
  using TestType = cl::sycl::vec<TypeParam, 2>;
  std::vector<TypeParam> a = {1, 1, 1, 1, 1, 1, 1, 1, 1,
                              2, 2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<TypeParam> b = {1, 1, 1, 2, 2, 2, 3, 3, 3,
                              1, 1, 1, 2, 2, 2, 3, 3, 3};
  std::vector<TypeParam> c = {1, 2, 3, 1, 2, 3, 1, 2, 3,
                              1, 2, 3, 1, 2, 3, 1, 2, 3};
  std::vector<TypeParam> exp = {2, 3, 4, 3, 4, 5, 4, 5, 6,
                                3, 4, 5, 5, 6, 7, 7, 8, 9};
  this->template check_mad_values<TestType>(a, b, c, exp);
}
TYPED_TEST(MathTest, Vector3DValues) {
  using TestType = cl::sycl::vec<TypeParam, 3>;
  std::vector<TypeParam> a = {1, 1, 1, 1, 1, 1, 1, 1, 1,
                              2, 2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<TypeParam> b = {1, 1, 1, 2, 2, 2, 3, 3, 3,
                              1, 1, 1, 2, 2, 2, 3, 3, 3};
  std::vector<TypeParam> c = {1, 2, 3, 1, 2, 3, 1, 2, 3,
                              1, 2, 3, 1, 2, 3, 1, 2, 3};
  std::vector<TypeParam> exp = {2, 3, 4, 3, 4, 5, 4, 5, 6,
                                3, 4, 5, 5, 6, 7, 7, 8, 9};
  this->template check_mad_values<TestType>(a, b, c, exp);
}
TYPED_TEST(MathTest, Vector4DValues) {
  using TestType = cl::sycl::vec<TypeParam, 4>;
  std::vector<TypeParam> a = {1, 1, 1, 1, 1, 1, 1, 1, 1,
                              2, 2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<TypeParam> b = {1, 1, 1, 2, 2, 2, 3, 3, 3,
                              1, 1, 1, 2, 2, 2, 3, 3, 3};
  std::vector<TypeParam> c = {1, 2, 3, 1, 2, 3, 1, 2, 3,
                              1, 2, 3, 1, 2, 3, 1, 2, 3};
  std::vector<TypeParam> exp = {2, 3, 4, 3, 4, 5, 4, 5, 6,
                                3, 4, 5, 5, 6, 7, 7, 8, 9};
  this->template check_mad_values<TestType>(a, b, c, exp);
}
TYPED_TEST(MathTest, Vector8DValues) {
  using TestType = cl::sycl::vec<TypeParam, 8>;
  std::vector<TypeParam> a = {1, 1, 1, 1, 1, 1, 1, 1, 1,
                              2, 2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<TypeParam> b = {1, 1, 1, 2, 2, 2, 3, 3, 3,
                              1, 1, 1, 2, 2, 2, 3, 3, 3};
  std::vector<TypeParam> c = {1, 2, 3, 1, 2, 3, 1, 2, 3,
                              1, 2, 3, 1, 2, 3, 1, 2, 3};
  std::vector<TypeParam> exp = {2, 3, 4, 3, 4, 5, 4, 5, 6,
                                3, 4, 5, 5, 6, 7, 7, 8, 9};
  this->template check_mad_values<TestType>(a, b, c, exp);
}
TYPED_TEST(MathTest, Vector16DValues) {
  using TestType = cl::sycl::vec<TypeParam, 16>;
  std::vector<TypeParam> a = {1, 1, 1, 1, 1, 1, 1, 1, 1,
                              2, 2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<TypeParam> b = {1, 1, 1, 2, 2, 2, 3, 3, 3,
                              1, 1, 1, 2, 2, 2, 3, 3, 3};
  std::vector<TypeParam> c = {1, 2, 3, 1, 2, 3, 1, 2, 3,
                              1, 2, 3, 1, 2, 3, 1, 2, 3};
  std::vector<TypeParam> exp = {2, 3, 4, 3, 4, 5, 4, 5, 6,
                                3, 4, 5, 5, 6, 7, 7, 8, 9};
  this->template check_mad_values<TestType>(a, b, c, exp);
}
