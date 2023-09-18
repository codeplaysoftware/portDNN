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

#include "portdnn/compat/convolution.hpp"
#include "test/gen/iota_initialised_data.h"

#include <type_traits>

using namespace sycldnn;
using namespace sycldnn::compat;

class DescriptorTest : public ::testing::Test {
 protected:
  void do_test(const std::vector<int>& in_sizes,    // nchw
               const std::vector<int>& filt_sizes,  // kchw
               const sycldnn::DataFormat format) {
    // setup tensor desc
    SNNDataType descDataType = SNNDataType::SNN_FLOAT;
    TensorDescriptor tensor_desc;
    setTensor4dDescriptor(tensor_desc, format, descDataType, in_sizes[0],
                          in_sizes[1], in_sizes[2], in_sizes[3]);
    int inN;
    int inC;
    int inH;
    int inW;
    int inStrideN;
    int inStrideC;
    int inStrideH;
    int inStrideW;
    getTensor4dDescriptor(tensor_desc, &descDataType, &inN, &inC, &inH, &inW,
                          &inStrideN, &inStrideC, &inStrideH, &inStrideW);

    // setup filter desc
    FilterDescriptor filter_desc;
    setFilter4dDescriptor(filter_desc, descDataType, format, filt_sizes[0],
                          filt_sizes[1], filt_sizes[2], filt_sizes[3]);
    sycldnn::FilterFormat filterFormat;
    int filterK;
    int filterC;
    int filterH;
    int filterW;
    getFilter4dDescriptor(filter_desc, &descDataType, &filterFormat, &filterK,
                          &filterC, &filterH, &filterW);

    // calculate out strides
    std::vector<int> strides;
    if (format == sycldnn::DataFormat::NCHW) {
      strides = {in_sizes[1] * in_sizes[2] * in_sizes[3],
                 in_sizes[2] * in_sizes[3], in_sizes[3], 1};
    } else if (format == sycldnn::DataFormat::NHWC) {
      strides = {in_sizes[1] * in_sizes[2] * in_sizes[3], 1,
                 in_sizes[3] * in_sizes[1], in_sizes[1]};
    } else
      FAIL() << "Unsuported Format!";

    // Confirm tensor desc set/get
    EXPECT_EQ(inN, in_sizes[0]);
    EXPECT_EQ(inC, in_sizes[1]);
    EXPECT_EQ(inH, in_sizes[2]);
    EXPECT_EQ(inW, in_sizes[3]);
    if (format == sycldnn::DataFormat::NCHW) {
      EXPECT_EQ(inStrideN, strides[0]);
      EXPECT_EQ(inStrideC, strides[1]);
      EXPECT_EQ(inStrideH, strides[2]);
      EXPECT_EQ(inStrideW, strides[3]);
    } else {  // NHWC
      EXPECT_EQ(inStrideN, strides[0]);
      EXPECT_EQ(inStrideH, strides[1]);
      EXPECT_EQ(inStrideW, strides[2]);
      EXPECT_EQ(inStrideC, strides[3]);
    }

    // Confirm filter desc set/get
    EXPECT_EQ(filterK, filt_sizes[0]);
    EXPECT_EQ(filterC, filt_sizes[1]);
    EXPECT_EQ(filterH, filt_sizes[2]);
    EXPECT_EQ(filterW, filt_sizes[3]);
    EXPECT_EQ(format == sycldnn::DataFormat::NCHW &&
                      filterFormat == sycldnn::FilterFormat::FCHW ||
                  format == sycldnn::DataFormat::NHWC &&
                      filterFormat == sycldnn::FilterFormat::HWCF,
              true);
  }
};

/**
 * Input:  1  2  3  4    Filter:  1  2  3
 *         5  6  7  8             4  5  6
 *         9 10 11 12             7  8  9
 *        13 14 15 16
 */
TEST_F(DescriptorTest, simple_3x3) {
  this->do_test({1, 1, 4, 4}, {1, 1, 3, 3}, sycldnn::DataFormat::NHWC);
}

/**
 * Input: 1   4    Filter: 1
 *         2   5            2
 *          3   6            3
 */

TEST_F(DescriptorTest, BatchedDeep1x1) {
  this->do_test({2, 3, 1, 1}, {1, 3, 1, 1}, sycldnn::DataFormat::NHWC);
}

/**
 * Input:       Filter: 1 2 3
 *         1            4 5 6
 *                      7 8 9
 *
 */
TEST_F(DescriptorTest, Simple1x1And3x3Filter) {
  this->do_test({1, 1, 1, 1}, {1, 1, 3, 3}, sycldnn::DataFormat::NHWC);
}

/**
 * Input: 1     Filter: 1
 *         2             2
 *          3             3
 */
TEST_F(DescriptorTest, Deep1x1) {
  this->do_test({1, 3, 1, 1}, {1, 3, 1, 1}, sycldnn::DataFormat::NHWC);
}

/**
 * Input:                 Filter: 1  10 19
 *          1                     4  13 22
 *                                7  16 25
 *
 *                                   2  11 20
 *            2                      5  14 23
 *                                   8  17 26
 *
 *                                      3  12 21
 *              3                       6  15 24
 *                                      9  18 27
 */
TEST_F(DescriptorTest, Deep1x1And3x3Filter) {
  this->do_test({1, 3, 1, 1}, {1, 3, 3, 3}, sycldnn::DataFormat::NHWC);
}

TEST_F(DescriptorTest, ForwardWindow3Stride1) {
  this->do_test({1, 1, 4, 4}, {1, 1, 3, 3}, sycldnn::DataFormat::NHWC);
}

TEST_F(DescriptorTest, ForwardWindow3Stride2VALID1x5x5x1x1) {
  this->do_test({1, 1, 5, 5}, {1, 1, 3, 3}, sycldnn::DataFormat::NCHW);
}

TEST_F(DescriptorTest, ForwardWindow7Stride4SAME1x11x11x1x2) {
  this->do_test({1, 1, 11, 11}, {2, 1, 7, 7}, sycldnn::DataFormat::NHWC);
}
