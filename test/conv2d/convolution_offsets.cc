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

#include "portdnn/backend/snn_backend.h"

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/params.h"

#include "portdnn/helpers/padding.h"
#include "portdnn/padding_mode.h"

#include "test/conv2d/convolution_fixture.h"
#include "test/conv2d/selector_list.h"

#include "test/types/cartesian_product.h"
#include "test/types/data_format_types.h"
#include "test/types/kernel_data_types.h"
#include "test/types/nested_pairs_to_tuple4.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"
#include "test/types/type_list.h"

#include <vector>

template <typename Tuple>
using OffsetConvolutionTest = ConvolutionFixture<Tuple>;

using DataTypeList = sycldnn::types::KernelDataTypes;
using Selectors = sycldnn::types::SelectorList;
using Backends = sycldnn::types::DefaultBackendTypes;
using DataFormats = sycldnn::types::DataFormatTypes;

using SNNTypePairs =
    sycldnn::types::CartesianProduct<Selectors, DataTypeList>::type;
using BackendTypePairs =
    sycldnn::types::CartesianProduct<SNNTypePairs, Backends>::type;
using DataFormatBackendTypePairs =
    sycldnn::types::CartesianProduct<BackendTypePairs, DataFormats>::type;
using TestTuple4 =
    sycldnn::types::NestedPairsToTuple4<DataFormatBackendTypePairs>::type;

using GTestTypeTuple4s = sycldnn::types::ToGTestTypes<TestTuple4>::type;
TYPED_TEST_SUITE(OffsetConvolutionTest, GTestTypeTuple4s);

sycldnn::conv2d::Conv2DParams get_3x3_params() {
  sycldnn::conv2d::Conv2DParams params;
  params.channels = 1;
  params.features = 1;
  params.batch = 1;
  params.in_rows = 4;
  params.in_cols = 4;
  params.window_rows = 3;
  params.window_cols = 3;
  params.stride_rows = 1;
  params.stride_cols = 1;
  params.out_rows = 2;
  params.out_cols = 2;
  params.pad_rows = 0;
  params.pad_cols = 0;
  params.dilation_rows = 1;
  params.dilation_cols = 1;
  return params;
}
sycldnn::conv2d::Conv2DParams get_3x3_stride2_params() {
  sycldnn::conv2d::Conv2DParams params;
  params.channels = 1;
  params.features = 1;
  params.batch = 1;
  params.in_rows = 4;
  params.in_cols = 4;
  params.window_rows = 3;
  params.window_cols = 3;
  params.stride_rows = 2;
  params.stride_cols = 2;
  params.out_rows = 2;
  params.out_cols = 2;
  params.pad_rows = 0;
  params.pad_cols = 0;
  params.dilation_rows = 1;
  params.dilation_cols = 1;
  return params;
}
sycldnn::conv2d::Conv2DParams get_1x1_params() {
  sycldnn::conv2d::Conv2DParams params;
  params.channels = 2;
  params.features = 2;
  params.batch = 1;
  params.in_rows = 3;
  params.in_cols = 3;
  params.window_rows = 1;
  params.window_cols = 1;
  params.stride_rows = 1;
  params.stride_cols = 1;
  params.out_rows = 3;
  params.out_cols = 3;
  params.pad_rows = 0;
  params.pad_cols = 0;
  params.dilation_rows = 1;
  params.dilation_cols = 1;
  return params;
}
sycldnn::conv2d::Conv2DParams get_5x5_params() {
  sycldnn::conv2d::Conv2DParams params;
  params.channels = 1;
  params.features = 2;
  params.batch = 1;
  params.in_rows = 7;
  params.in_cols = 7;
  params.window_rows = 5;
  params.window_cols = 5;
  params.stride_rows = 2;
  params.stride_cols = 2;
  params.dilation_rows = 1;
  params.dilation_cols = 1;
  return params;
}

TYPED_TEST(OffsetConvolutionTest, Simple3x3) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {348, 393, 528, 573};
  auto params = get_3x3_params();
  this->template test_conv<sycldnn::conv2d::conv_type::Forward>(exp, params, 0,
                                                                128, 32, 64);
}
TYPED_TEST(OffsetConvolutionTest, Simple1x1) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {7,  10, 15, 22, 23, 34, 31, 46, 39,
                               58, 47, 70, 55, 82, 63, 94, 71, 106};
  auto params = get_1x1_params();
  this->template test_conv<sycldnn::conv2d::conv_type::Forward>(exp, params, 0,
                                                                32, 128, 64);
}
TYPED_TEST(OffsetConvolutionTest, InputBackprop3x3) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {1,  4,  7,  6,  7,  23, 33, 24,
                               19, 53, 63, 42, 21, 52, 59, 36};
  auto params = get_3x3_params();
  this->template test_conv<sycldnn::conv2d::conv_type::InputBackprop>(
      exp, params, 0, 128, 32, 64);
}
TYPED_TEST(OffsetConvolutionTest, InputBackprop3x3Stride2) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {1,  2,  5,  4,  4,  5,  14, 10,
                               10, 14, 36, 24, 12, 15, 34, 20};
  auto params = get_3x3_stride2_params();
  this->template test_conv<sycldnn::conv2d::conv_type::InputBackprop>(
      exp, params, 0, 128, 32, 64);
}
TYPED_TEST(OffsetConvolutionTest, InputBackprop1x1) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {5,  11, 11, 25, 17, 39, 23,  53, 29,
                               67, 35, 81, 41, 95, 47, 109, 53, 123};
  auto params = get_1x1_params();
  this->template test_conv<sycldnn::conv2d::conv_type::InputBackprop>(
      exp, params, 0, 128, 32, 64);
}
TYPED_TEST(OffsetConvolutionTest, FilterBackprop3x3) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {44, 54, 64, 84, 94, 104, 124, 134, 144};
  auto params = get_3x3_params();
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(
      exp, params, 0, 128, 32, 64);
}
TYPED_TEST(OffsetConvolutionTest, FilterBackprop1x1) {
  using DataType = typename TestFixture::DataType;
  std::vector<DataType> exp = {969, 1050, 1050, 1140};
  auto params = get_1x1_params();
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(
      exp, params, 0, 128, 32, 64);
}
TYPED_TEST(OffsetConvolutionTest, ForwardSAME1x7x7x1x2) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {
      3429.,  3510.,  6010.,  6160.,  7060.,  7240.,  4293.,  4410.,
      8600.,  8840.,  14225., 14650., 15475., 15950., 9020.,  9320.,
      14270., 14720., 22975., 23750., 24225., 25050., 13850., 14360.,
      6093.,  6426.,  9310.,  9880.,  9760.,  10360., 5229.,  5598.};
  const auto padding = sycldnn::PaddingMode::SAME;
  auto params = get_5x5_params();
  params = sycldnn::helpers::add_padding_to(params, padding);
  const DataType max_input_val = 2048.0;
  this->template test_conv<sycldnn::conv2d::conv_type::Forward>(
      exp_out, params, max_input_val, 32, 32, 32);
}
TYPED_TEST(OffsetConvolutionTest, FilterBackpropSAME1x7x7x1x2) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {
      3909., 4062., 4098., 4260., 5276., 5492., 3774., 3936., 3945., 4116.,
      5232., 5448., 5421., 5646., 6956., 7256., 4971., 5196., 5142., 5376.,
      6608., 6896., 6812., 7112., 8720., 9120., 6212., 6512., 6392., 6704.,
      3504., 3720., 3621., 3846., 4556., 4856., 3171., 3396., 3270., 3504.,
      4323., 4602., 4440., 4728., 5564., 5948., 3864., 4152., 3963., 4260.};
  const auto padding = sycldnn::PaddingMode::SAME;
  auto params = get_5x5_params();
  params = sycldnn::helpers::add_padding_to(params, padding);
  const DataType max_input_val = 2048.0;
  this->template test_conv<sycldnn::conv2d::conv_type::FilterBackprop>(
      exp_out, params, max_input_val, 32, 32, 32);
}
TYPED_TEST(OffsetConvolutionTest, InputBackpropSAME1x7x7x1x2) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {
      368.,  472.,  854.,  720.,  1226., 968.,  1136.,  888.,  992.,  1754.,
      1400., 2366., 1808., 1976., 1660., 1912., 3267.,  2524., 4185., 3136.,
      3484., 2392., 2624., 4202., 3032., 4814., 3440.,  3736., 3916., 4360.,
      6939., 4972., 7857., 5584., 6124., 3896., 4256.,  6650., 4664., 7262.,
      5072., 5496., 5696., 6056., 9470., 6624., 10322., 7192., 7616.};
  const auto padding = sycldnn::PaddingMode::SAME;
  auto params = get_5x5_params();
  params = sycldnn::helpers::add_padding_to(params, padding);
  const DataType max_input_val = 2048.0;
  this->template test_conv<sycldnn::conv2d::conv_type::InputBackprop>(
      exp_out, params, max_input_val, 32, 32, 32);
}
