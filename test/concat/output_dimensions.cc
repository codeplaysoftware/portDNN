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

#include "sycldnn/concat/launch.h"
#include "sycldnn/concat/params.h"

struct ConcatOutputDimsTestFixture : public ::testing::Test {
 public:
  ConcatOutputDimsTestFixture() {}
  void test_concat_output_dims(std::vector<int> exp,
                               sycldnn::concat::ConcatParams& params) {
    auto output_dims = sycldnn::concat::calculate_concat_output_dims(params);

    EXPECT_EQ(exp.size(), output_dims.size());
    for (size_t i = 0; i < output_dims.size(); i++) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      EXPECT_EQ(exp[i], output_dims[i]);
    }
  }
};

TEST_F(ConcatOutputDimsTestFixture, output_dims_rank_1) {
  sycldnn::concat::ConcatParams params;
  params.rank = 1;
  params.axis = 0;
  params.input_descriptors = {
      {1},  {2},  {3},  {4},  {5},  {6},  {7},  {8},  {9},  {10}, {11}, {12},
      {13}, {14}, {15}, {16}, {17}, {18}, {19}, {20}, {21}, {22}, {23}, {24},
      {25}, {26}, {27}, {28}, {29}, {30}, {31}, {32}, {33}, {34}, {35}};
  std::vector<int> exp_out = {630};
  this->test_concat_output_dims(exp_out, params);
}

TEST_F(ConcatOutputDimsTestFixture, output_dims_rank_2) {
  sycldnn::concat::ConcatParams params;
  params.rank = 2;
  params.axis = 1;
  params.input_descriptors = {
      {4, 1},  {4, 2},  {4, 3},  {4, 4},  {4, 5},  {4, 6},  {4, 7},
      {4, 8},  {4, 9},  {4, 10}, {4, 11}, {4, 12}, {4, 13}, {4, 14},
      {4, 15}, {4, 16}, {4, 17}, {4, 18}, {4, 19}, {4, 20}, {4, 21},
      {4, 22}, {4, 23}, {4, 24}, {4, 25}, {4, 26}, {4, 27}, {4, 28},
      {4, 29}, {4, 30}, {4, 31}, {4, 32}, {4, 33}, {4, 34}};
  std::vector<int> exp_out = {4, 595};
  this->test_concat_output_dims(exp_out, params);
}

TEST_F(ConcatOutputDimsTestFixture, output_dims_rank_3) {
  sycldnn::concat::ConcatParams params;
  params.rank = 3;
  params.axis = 2;
  params.input_descriptors = {{3, 3, 1}, {3, 3, 2}, {3, 3, 3}, {3, 3, 4},
                              {3, 3, 5}, {3, 3, 6}, {3, 3, 7}, {3, 3, 8},
                              {3, 3, 9}, {3, 3, 10}};
  std::vector<int> exp_out = {3, 3, 55};
  this->test_concat_output_dims(exp_out, params);
}

TEST_F(ConcatOutputDimsTestFixture, output_dims_rank_4) {
  sycldnn::concat::ConcatParams params;
  params.rank = 4;
  params.axis = 3;
  params.input_descriptors = {
      {2, 2, 2, 1},  {2, 2, 2, 2},  {2, 2, 2, 3},  {2, 2, 2, 4},  {2, 2, 2, 5},
      {2, 2, 2, 6},  {2, 2, 2, 7},  {2, 2, 2, 8},  {2, 2, 2, 9},  {2, 2, 2, 10},
      {2, 2, 2, 11}, {2, 2, 2, 12}, {2, 2, 2, 13}, {2, 2, 2, 14}, {2, 2, 2, 15},
      {2, 2, 2, 16}, {2, 2, 2, 17}, {2, 2, 2, 18}, {2, 2, 2, 19}, {2, 2, 2, 20},
      {2, 2, 2, 21}, {2, 2, 2, 22}, {2, 2, 2, 23}, {2, 2, 2, 24}, {2, 2, 2, 25},
      {2, 2, 2, 26}, {2, 2, 2, 27}, {2, 2, 2, 28}, {2, 2, 2, 29}, {2, 2, 2, 30},
      {2, 2, 2, 31}, {2, 2, 2, 32}, {2, 2, 2, 33}, {2, 2, 2, 34}};
  std::vector<int> exp_out = {2, 2, 2, 595};
  this->test_concat_output_dims(exp_out, params);
}

TEST_F(ConcatOutputDimsTestFixture, output_dims_negative_axis) {
  sycldnn::concat::ConcatParams params;
  params.rank = 4;
  params.axis = -1;
  params.input_descriptors = {
      {2, 2, 2, 1},  {2, 2, 2, 2},  {2, 2, 2, 3},  {2, 2, 2, 4},  {2, 2, 2, 5},
      {2, 2, 2, 6},  {2, 2, 2, 7},  {2, 2, 2, 8},  {2, 2, 2, 9},  {2, 2, 2, 10},
      {2, 2, 2, 11}, {2, 2, 2, 12}, {2, 2, 2, 13}, {2, 2, 2, 14}, {2, 2, 2, 15},
      {2, 2, 2, 16}, {2, 2, 2, 17}, {2, 2, 2, 18}, {2, 2, 2, 19}, {2, 2, 2, 20},
      {2, 2, 2, 21}, {2, 2, 2, 22}, {2, 2, 2, 23}, {2, 2, 2, 24}, {2, 2, 2, 25},
      {2, 2, 2, 26}, {2, 2, 2, 27}, {2, 2, 2, 28}, {2, 2, 2, 29}, {2, 2, 2, 30},
      {2, 2, 2, 31}, {2, 2, 2, 32}, {2, 2, 2, 33}, {2, 2, 2, 34}};
  std::vector<int> exp_out = {2, 2, 2, 595};
  this->test_concat_output_dims(exp_out, params);
}