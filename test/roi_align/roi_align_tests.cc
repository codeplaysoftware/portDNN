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
#include "test/types/cartesian_product.h"
#include "test/types/kernel_data_types.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"

#include "test/roi_align/roi_align_fixture.h"

#include <array>
#include <vector>

using namespace sycldnn;  // NOLINT(google-build-using-namespace)
using DataTypeList = sycldnn::types::KernelDataTypes;
using Backends = sycldnn::types::DefaultBackendTypes;

using SNNTypePairs =
    sycldnn::types::CartesianProduct<DataTypeList, Backends>::type;
using GTestTypePairs = sycldnn::types::ToGTestTypes<SNNTypePairs>::type;

template <typename Pair>
using RoiAlignTestFixture =
    RoiAlignFixture<typename Pair::FirstType, /*batch indices type*/ int32_t,
                    typename Pair::SecondType>;
template <typename Pair>
using RoiAlignTestFixtureInt64 =
    RoiAlignFixture<typename Pair::FirstType, /*batch indices type*/ int64_t,
                    typename Pair::SecondType>;

TYPED_TEST_SUITE(RoiAlignTestFixture, GTestTypePairs);
TYPED_TEST_SUITE(RoiAlignTestFixtureInt64, GTestTypePairs);
TYPED_TEST(RoiAlignTestFixture, 2x2_TO_1x1_DEFAULT_PARAMS) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out_max_pool = {0.75};
  const std::vector<DataType> exp_out_avg_pool = {1.5};
  const std::vector<DataType> rois = {0., 0., 1., 1.};
  const std::vector<int32_t> batch_indices = {0};
  roi_align::RoiAlignParams params;
  params.batch = 1;
  params.channels = 1;
  params.in_height = 2;
  params.in_width = 2;
  params.out_height = 1;
  params.out_width = 1;
  params.num_rois = 1;
  params.sampling_ratio = 0;
  params.spatial_scale = 1.0f;
  this->test_roi_align(rois, batch_indices, exp_out_max_pool, exp_out_avg_pool,
                       params);
}
TYPED_TEST(RoiAlignTestFixture, 2x2_TO_1x1_DEFAULT_PARAMS_FLIPPED) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out_max_pool = {3.};
  const std::vector<DataType> exp_out_avg_pool = {3.};
  const std::vector<DataType> rois = {1., 1., 0., 0.};
  const std::vector<int32_t> batch_indices = {0};
  roi_align::RoiAlignParams params;
  params.batch = 1;
  params.channels = 1;
  params.in_height = 2;
  params.in_width = 2;
  params.out_height = 1;
  params.out_width = 1;
  params.num_rois = 1;
  params.sampling_ratio = 0;
  params.spatial_scale = 1.0f;
  this->test_roi_align(rois, batch_indices, exp_out_max_pool, exp_out_avg_pool,
                       params);
}
TYPED_TEST(RoiAlignTestFixture, 2x2_TO_3x3_DEFAULT_PARAMS) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out_max_pool = {
      0.2777778, 0.41666666, 0.6944445, 0.8333333, 0.75,
      1.25,      1.388889,   1.25,      2.0833335};
  const std::vector<DataType> exp_out_avg_pool = {
      0.5,       0.8333333, 1.1666667, 1.1666666, 1.5,
      1.8333333, 1.8333335, 2.1666667, 2.5};
  const std::vector<DataType> rois = {0., 0., 1., 1.};
  const std::vector<int32_t> batch_indices = {0};
  roi_align::RoiAlignParams params;
  params.batch = 1;
  params.channels = 1;
  params.in_height = 2;
  params.in_width = 2;
  params.out_height = 3;
  params.out_width = 3;
  params.num_rois = 1;
  params.sampling_ratio = 0;
  params.spatial_scale = 1.0f;
  this->test_roi_align(rois, batch_indices, exp_out_max_pool, exp_out_avg_pool,
                       params);
}
TYPED_TEST(RoiAlignTestFixture, 2x2_TO_3x3_DEFAULT_PARAMS_FLIPPED) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out_max_pool = {3., 3., 3., 3., 3.,
                                                  3., 3., 3., 3.};
  const std::vector<DataType> exp_out_avg_pool = {3., 3., 3., 3., 3.,
                                                  3., 3., 3., 3.};
  const std::vector<DataType> rois = {1., 1., 0., 0.};
  const std::vector<int32_t> batch_indices = {0};
  roi_align::RoiAlignParams params;
  params.batch = 1;
  params.channels = 1;
  params.in_height = 2;
  params.in_width = 2;
  params.out_height = 3;
  params.out_width = 3;
  params.num_rois = 1;
  params.sampling_ratio = 0;
  params.spatial_scale = 1.0f;
  this->test_roi_align(rois, batch_indices, exp_out_max_pool, exp_out_avg_pool,
                       params);
}
TYPED_TEST(RoiAlignTestFixture, 3x3_TO_2x2_TWO_BOXES) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out_max_pool = {0.5625, 0.75, 1.6875, 2.25,
                                                  0.5625, 0.75, 1.6875, 2.25};
  const std::vector<DataType> exp_out_avg_pool = {1., 1.5, 2.5, 3.,
                                                  1., 1.5, 2.5, 3.};
  const std::vector<DataType> rois = {0., 0., 1., 1., 0., 0., 0.5, 0.5};
  const std::vector<int32_t> batch_indices = {0, 0};
  roi_align::RoiAlignParams params;
  params.batch = 1;
  params.channels = 1;
  params.in_height = 3;
  params.in_width = 3;
  params.out_height = 2;
  params.out_width = 2;
  params.num_rois = 2;
  params.sampling_ratio = 0;
  params.spatial_scale = 1.0f;
  this->test_roi_align(rois, batch_indices, exp_out_max_pool, exp_out_avg_pool,
                       params);
}
TYPED_TEST(RoiAlignTestFixture, 3x3_TO_2x2_TWO_BOXES_FLIPPED) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out_max_pool = {2.25, 2.8125, 3.9375, 4.5,
                                                  2.25, 2.25,   2.25,   2.25};
  const std::vector<DataType> exp_out_avg_pool = {5., 5.5, 6.5, 7.,
                                                  3., 3.5, 4.5, 5.};
  const std::vector<DataType> rois = {1., 1., 0., 0., 0.5, 0.5, 0., 0.};
  const std::vector<int32_t> batch_indices = {0, 0};
  roi_align::RoiAlignParams params;
  params.batch = 1;
  params.channels = 1;
  params.in_height = 3;
  params.in_width = 3;
  params.out_height = 2;
  params.out_width = 2;
  params.num_rois = 2;
  params.sampling_ratio = 0;
  params.spatial_scale = 1.0f;
  this->test_roi_align(rois, batch_indices, exp_out_max_pool, exp_out_avg_pool,
                       params);
}
TYPED_TEST(RoiAlignTestFixture, 2x2_TO_3x3_EXTRAPOLATED) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out_max_pool = {0.,
                                                  0.00000002980232,
                                                  0.66666675,
                                                  0.00000005960464,
                                                  0.00000005960464,
                                                  0.66666675,
                                                  1.3333335,
                                                  1.3333335,
                                                  1.3333336};
  const std::vector<DataType> exp_out_avg_pool = {0.,
                                                  0.00000002980232,
                                                  0.66666675,
                                                  0.00000005960464,
                                                  0.00000008940697,
                                                  0.6666668,
                                                  1.3333335,
                                                  1.3333336,
                                                  2.0000002};
  const std::vector<DataType> rois = {-1., -1., 1., 1.};
  const std::vector<int32_t> batch_indices = {0};
  roi_align::RoiAlignParams params;
  params.batch = 1;
  params.channels = 1;
  params.in_height = 2;
  params.in_width = 2;
  params.out_height = 3;
  params.out_width = 3;
  params.num_rois = 1;
  params.sampling_ratio = 0;
  params.spatial_scale = 1.0f;
  this->test_roi_align(rois, batch_indices, exp_out_max_pool, exp_out_avg_pool,
                       params);
}
TYPED_TEST(RoiAlignTestFixture, 5x5_TO_3x4_FOUR_BOXES_PARAMS) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> rois = {7.,   5.,   7.,   5.,   -15., -15., -15.,
                                      -15., -10., 21.,  -10., 21.,  13.,  8.,
                                      13.,  8.,   -14., 19.,  -14., 19.};
  const std::vector<int32_t> batch_indices = {0, 0, 0, 0, 0};
  const std::vector<DataType> exp_out_max_pool = {
      2.109375,   2.953125,   3.375,     2.53125,   3.3593752, 4.703125,
      5.375,      4.03125,    3.515625,  4.921875,  5.625,     4.21875,
      10.8984375, 15.2578125, 17.4375,   13.078125, 17.356771, 24.29948,
      27.770834,  20.828125,  18.164062, 25.429688, 29.0625,   21.796875,
      19.6875,    27.5625,    31.5,      23.625,    31.354168, 43.895836,
      50.166668,  37.625,     32.8125,   45.9375,   52.5,      39.375,
      0.,         0.,         0.,        0.,        0.,        0.,
      0.,         0.,         0.,        0.,        0.,        0.,
      25.,        25.,        25.,       25.,       25.,       25.,
      25.,        25.,        25.,       25.,       25.,       25.,
      50.,        50.,        50.,       50.,       50.,       50.,
      50.,        50.,        50.,       50.,       50.,       50.,
      5.625,      5.625,      5.625,     4.5703125, 8.958334,  8.958334,
      8.958334,   7.2786465,  9.375,     9.375,     9.375,     7.6171875,
      19.6875,    19.6875,    19.6875,   15.996094, 31.354168, 31.354168,
      31.354168,  25.475262,  32.8125,   32.8125,   32.8125,   26.660156,
      33.75,      33.75,      33.75,     27.421875, 53.750004, 53.750004,
      53.750004,  43.67188,   56.25,     56.25,     56.25,     45.703125,
      4.5,        3.9375,     2.8125,    3.9375,    5.5,       4.8125,
      3.4375,     4.8125,     4.583334,  4.0104175, 2.864584,  3.937499,
      23.25,      20.34375,   14.53125,  18.,       28.416668, 24.864584,
      17.760418,  22.,        23.249996, 20.343746, 14.531248, 17.999996,
      42.,        36.75,      26.25,     32.0625,   51.333336, 44.916668,
      32.083336,  39.1875,    41.999992, 36.749992, 26.249996, 32.062492,
      4.375,      4.375,      4.375,     4.375,     7.708334,  7.708334,
      7.708334,   7.708334,   9.375001,  9.375001,  9.375001,  9.375001,
      21.874998,  21.874998,  21.874998, 21.874998, 26.979168, 26.979168,
      26.979168,  26.979168,  32.812504, 32.812504, 32.812504, 32.812504,
      40.104164,  40.104164,  40.104164, 40.104164, 46.250004, 46.250004,
      46.250004,  46.250004,  56.250008, 56.250008, 56.250008, 56.250008};
  const std::vector<DataType> exp_out_avg_pool = {
      2.9583333, 3.2083333, 3.4583333, 3.7083333, 4.625,     4.875,
      5.125,     5.375,     6.291667,  6.541667,  6.791667,  7.041667,
      27.958334, 28.208334, 28.458332, 28.708332, 29.624998, 29.875,
      30.125,    30.374998, 31.291668, 31.541668, 31.791666, 32.041664,
      52.958332, 53.208332, 53.458332, 53.708332, 54.625,    54.875004,
      55.125004, 55.375,    56.29167,  56.541664, 56.79167,  57.04167,
      0.,        0.,        0.,        0.,        0.,        0.,
      0.,        0.,        0.,        0.,        0.,        0.,
      25.,       25.,       25.,       25.,       25.,       25.,
      25.,       25.,       25.,       25.,       25.,       25.,
      50.,       50.,       50.,       50.,       50.,       50.,
      50.,       50.,       50.,       50.,       50.,       50.,
      7.3958335, 7.3958335, 7.4270835, 7.6458335, 9.062501,  9.062501,
      9.093751,  9.312501,  10.729167, 10.729167, 10.760417, 10.979168,
      32.395832, 32.395832, 32.427082, 32.645836, 34.062504, 34.062504,
      34.093754, 34.3125,   35.729168, 35.729168, 35.760418, 35.979164,
      57.395832, 57.395832, 57.427082, 57.645832, 59.0625,   59.0625,
      59.093754, 59.312504, 60.72917,  60.72917,  60.760418, 60.97917,
      4.270833,  4.5208335, 4.7708335, 5.0208335, 5.9375,    6.1875,
      6.4375,    6.6875,    7.604167,  7.854167,  8.104168,  8.354167,
      29.270832, 29.520832, 29.770834, 30.020834, 30.9375,   31.187498,
      31.4375,   31.6875,   32.604168, 32.854164, 33.104168, 33.354164,
      54.270832, 54.520832, 54.770832, 55.020832, 55.937504, 56.1875,
      56.437504, 56.6875,   57.60417,  57.85417,  58.104168, 58.35417,
      6.7708335, 6.7708335, 6.7708335, 6.8020835, 8.437501,  8.437501,
      8.437501,  8.468751,  10.104166, 10.104166, 10.104166, 10.135416,
      31.770832, 31.770832, 31.770832, 31.802082, 33.437504, 33.437504,
      33.437504, 33.468754, 35.104168, 35.104168, 35.104168, 35.135418,
      56.770832, 56.770832, 56.770832, 56.802082, 58.4375,   58.4375,
      58.4375,   58.46875,  60.10417,  60.10417,  60.10417,  60.135418};
  roi_align::RoiAlignParams params;
  params.batch = 1;
  params.channels = 3;
  params.in_height = 5;
  params.in_width = 5;
  params.out_height = 3;
  params.out_width = 4;
  params.num_rois = 5;
  params.sampling_ratio = 2;
  params.spatial_scale = (1.f / 16.f);
  this->test_roi_align(rois, batch_indices, exp_out_max_pool, exp_out_avg_pool,
                       params);
}
TYPED_TEST(RoiAlignTestFixture, 5x5_TO_2x1_THREE_BOXES_BATCH_INDICES) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out_max_pool = {
      1.6875, 3.9375, 17.0625, 20.34375, 50.,    34.375,
      75.,    50.,    66.25,   92.75,    81.875, 114.625};
  const std::vector<DataType> exp_out_avg_pool = {
      1.75,    4.25,  26.75,  29.25,  50.5625, 52.75,
      75.5625, 77.75, 103.25, 105.75, 128.25,  130.75};
  const std::vector<DataType> rois = {0., 0., 0.5, 0.5, -1., -1.,
                                      1., 1., 1.,  1.,  0.,  0.5};
  const std::vector<int32_t> batch_indices = {0, 1, 2};
  roi_align::RoiAlignParams params;
  params.batch = 3;
  params.channels = 2;
  params.in_height = 5;
  params.in_width = 5;
  params.out_height = 2;
  params.out_width = 1;
  params.num_rois = 3;
  params.sampling_ratio = 2;
  params.spatial_scale = (1.f / 4.f);
  this->test_roi_align(rois, batch_indices, exp_out_max_pool, exp_out_avg_pool,
                       params);
}
TYPED_TEST(RoiAlignTestFixtureInt64, 11x13_TO_3x5_INT64) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out_max_pool = {
      6.63,      5.1566663,  3.9666667, 5.5533338,  7.14,      10.53,
      8.19,      6.3,        8.820001,  11.340001,  8.969999,  6.9766665,
      5.366667,  7.513334,   9.66,      79.56,      61.879997, 44.483334,
      62.276672, 80.07,      126.36,    98.28,      70.65,     98.91001,
      127.17001, 107.63999,  83.71999,  60.18333,   84.25667,  108.33,
      152.48999, 118.603325, 85.,       119.000015, 153.,      242.19,
      188.37,    135.,       189.00002, 243.00002,  206.30998, 160.46332,
      115.,      161.,       207.,      0.,         0.,        0.,
      0.,        0.,         0.,        429.,       429.,      265.98,
      404.19998, 0.,         429.,      429.,       265.98,    404.19998,
      0.,        0.,         0.,        0.,         0.,        0.,
      572.,      572.,       354.64,    538.61993,  0.,        572.,
      572.,      354.64,     538.61993, 0.,         0.,        0.,
      0.,        0.,         0.,        715.,       715.,      443.30002,
      673.04,    0.,         715.,      715.,       443.30002, 673.04,
      0.,        0.,         0.,        0.,         0.,        1.8000002,
      1.4000001, 1.5,        2.0999994, 2.6999996,  6.750001,  5.2500005,
      4.,        5.5999985,  7.199999,  0.,         0.,        0.,
      0.,        0.,         130.50002, 101.50001,  73.,       102.199974,
      131.39998, 71.100006,  55.300003, 39.75,      55.649986, 71.54999,
      0.,        0.,         0.,        0.,         0.,        259.2,
      201.6,     144.5,      202.29994, 260.09995,  135.45001, 105.350006,
      75.5,      105.699974, 135.89998};
  const std::vector<DataType> exp_out_avg_pool = {
      7.466667,  7.6666665, 7.866667,  8.066667,  8.266666,  11.8,
      12.,       12.200001, 12.400001, 12.6,      16.133333, 16.333334,
      16.533335, 16.733334, 16.933332, 150.46666, 150.66667, 150.86667,
      151.06667, 151.26666, 154.8,     155.,      155.20001, 155.4,
      155.6,     159.13332, 159.33334, 159.53333, 159.73334, 159.93332,
      293.46664, 293.66666, 293.86667, 294.06665, 294.26666, 297.8,
      298.,      298.2,     298.4,     298.6,     302.13333, 302.3333,
      302.53333, 302.73334, 302.93335, 0.,        0.,        0.,
      0.,        0.,        0.,        429.,      429.,      429.38,
      430.06,    0.,        429.,      429.,      429.38,    430.06,
      0.,        0.,        0.,        0.,        0.,        0.,
      572.,      572.,      572.38,    573.06,    0.,        572.,
      572.,      572.38,    573.06,    0.,        0.,        0.,
      0.,        0.,        0.,        715.,      715.,      715.38,
      716.06,    0.,        715.,      715.,      715.38,    716.06,
      0.,        0.,        0.,        0.,        0.,        2.1,
      2.3,       2.5,       2.6999998, 2.8999999, 8.6,       8.8,
      9.,        9.2,       9.4,       0.,        0.,        0.,
      0.,        0.,        145.1,     145.3,     145.5,     145.7,
      145.9,     151.6,     151.8,     152.,      152.20001, 152.4,
      0.,        0.,        0.,        0.,        0.,        288.09998,
      288.3,     288.5,     288.7,     288.9,     294.59998, 294.8,
      295.,      295.2,     295.4};
  const std::vector<DataType> rois = {0.,  0.2, 0.2, 0.4, -1., -1.,
                                      0.7, 0.1, 1.,  -1., 0.,  0.5};
  const std::vector<int64_t> batch_indices = {0, 1, 0};
  roi_align::RoiAlignParams params;
  params.batch = 4;
  params.channels = 3;
  params.in_height = 11;
  params.in_width = 13;
  params.out_height = 3;
  params.out_width = 5;
  params.num_rois = 3;
  params.sampling_ratio = 0;
  params.spatial_scale = 2.f;
  this->test_roi_align(rois, batch_indices, exp_out_max_pool, exp_out_avg_pool,
                       params);
}
TYPED_TEST(RoiAlignTestFixtureInt64, 3x4_TO_5x5_INT64) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out_max_pool = {
      0.,        36.,       36.,       36.,        36.,        0.,
      32.399998, 32.399998, 32.399998, 32.399998,  0.,         25.199999,
      25.199999, 25.199999, 25.199999, 0.,         20.000002,  20.000002,
      20.000002, 20.000002, 0.,        28.000002,  28.000002,  28.000002,
      28.000002, 0.,        48.,       48.,        48.,        48.,
      0.,        43.199997, 43.199997, 43.199997,  43.199997,  0.,
      33.6,      33.6,      33.6,      33.6,       0.,         26.000004,
      26.000004, 26.000004, 26.000004, 0.,         36.4,       36.4,
      36.4,      36.4,      0.,        60.,        60.,        60.,
      60.,       0.,        54.,       54.,        54.,        54.,
      0.,        42.,       42.,       42.,        42.,        0.,
      32.000004, 32.000004, 32.000004, 32.000004,  0.,         44.800003,
      44.800003, 44.800003, 44.800003, 0.5504,     0.42239997, 0.45360005,
      0.6216001, 0.7896001, 1.2384,    0.95039994, 0.972,      1.332,
      1.692,     1.9264001, 1.4783999, 1.5120001,  2.0720003,  2.6320002,
      2.6144001, 2.0064,    2.052,     2.8120005,  3.5720005,  3.3024,
      2.5344,    2.592,     3.5520005, 4.5120006,  8.6688,     6.6527996,
      5.8968005, 8.080801,  10.264801, 6.6048,     5.0687995,  4.4928,
      6.1568003, 7.820801,  7.7056003, 5.9135995,  5.1408005,  7.0448008,
      8.9488,    10.457601, 8.0256,    6.9768004,  9.5608015,  12.144802,
      13.2096,   10.1376,   8.8128,    12.076801,  15.340801,  17.3376,
      13.305599, 11.340001, 15.540002, 19.740002,  13.2096,    10.137599,
      8.64,      11.840001, 15.040001, 13.4848,    10.348799,  8.769601,
      12.017601, 15.265601, 18.3008,   14.044801,  11.901601,  16.309603,
      20.717604, 23.1168,   17.740799, 15.033601,  20.601603,  26.169601};
  const std::vector<DataType> exp_out_avg_pool = {
      0.,        36.,       36.,       36.,        36.,        0.,
      36.399998, 36.399998, 36.399998, 36.399998,  0.,         37.199997,
      37.199997, 37.199997, 37.199997, 0.,         38.,        38.,
      38.,       38.,       0.,        38.8,       38.8,       38.8,
      38.8,      0.,        48.,       48.,        48.,        48.,
      0.,        48.399998, 48.399998, 48.399998,  48.399998,  0.,
      49.199997, 49.199997, 49.199997, 49.199997,  0.,         50.,
      50.,       50.,       50.,       0.,         50.8,       50.8,
      50.8,      50.8,      0.,        60.,        60.,        60.,
      60.,       0.,        60.4,      60.4,       60.4,       60.4,
      0.,        61.2,      61.2,      61.2,       61.2,       0.,
      62.,       62.,       62.,       62.,        0.,         62.8,
      62.8,      62.8,      62.8,      0.78000003, 0.97999996, 1.1800001,
      1.3800001, 1.5800002, 1.5799999, 1.78,       1.9799999,  2.1799998,
      2.38,      2.38,      2.58,      2.78,       2.98,       3.18,
      3.1800003, 3.38,      3.58,      3.7800002,  3.9800003,  3.98,
      4.18,      4.38,      4.5800004, 4.78,       12.780001,  12.98,
      13.18,     13.380001, 13.58,     13.58,      13.779999,  13.9800005,
      14.179999, 14.379999, 14.38,     14.579999,  14.780001,  14.9800005,
      15.18,     15.18,     15.38,     15.58,      15.780001,  15.980001,
      15.98,     16.18,     16.380001, 16.58,      16.78,      24.78,
      24.98,     25.180002, 25.38,     25.58,      25.58,      25.779999,
      25.98,     26.18,     26.380001, 26.38,      26.58,      26.779999,
      26.98,     27.18,     27.179998, 27.38,      27.58,      27.78,
      27.980001, 27.98,     28.179996, 28.38,      28.58,      28.779999};
  const std::vector<DataType> rois = {-6., -1., 0.1, 0.2, 0.2, 0.3, -1., -2.};
  const std::vector<int64_t> batch_indices = {1, 0};
  roi_align::RoiAlignParams params;
  params.batch = 2;
  params.channels = 3;
  params.in_height = 3;
  params.in_width = 4;
  params.out_height = 5;
  params.out_width = 5;
  params.num_rois = 2;
  params.sampling_ratio = 1;
  params.spatial_scale = 0.2;
  this->test_roi_align(rois, batch_indices, exp_out_max_pool, exp_out_avg_pool,
                       params);
}
TYPED_TEST(RoiAlignTestFixtureInt64, 7x11_TO_2x1_INT64) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out_max_pool = {
      603.56256, 821.34, 653.80505, 888.9075, 704.04755, 956.475};
  const std::vector<DataType> exp_out_avg_pool = {
      929.0501, 934.55, 1006.05005, 1011.55005, 1083.05, 1088.55};
  const std::vector<DataType> rois = {0.2, 0.2, 0.3, 0.3};
  const std::vector<int64_t> batch_indices = {4};
  roi_align::RoiAlignParams params;
  params.batch = 5;
  params.channels = 3;
  params.in_height = 7;
  params.in_width = 11;
  params.out_height = 2;
  params.out_width = 1;
  params.num_rois = 1;
  params.sampling_ratio = 2;
  params.spatial_scale = 0.75f;
  this->test_roi_align(rois, batch_indices, exp_out_max_pool, exp_out_avg_pool,
                       params);
}
TYPED_TEST(RoiAlignTestFixtureInt64, 10x10_TO_4x2_INT64) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out_max_pool = {
      194.0039,  168.69531, 273.33984, 237.65625, 169.27734, 147.16406,
      244.51172, 212.57031, 258.45703, 224.55469, 361.23047, 313.82812,
      222.01172, 192.86719, 320.6836,  278.58594, 322.91016, 280.41406,
      449.1211,  390.,      274.7461,  238.57031, 396.85547, 344.60156};
  const std::vector<DataType> exp_out_avg_pool = {
      303.75, 305., 310., 311.25, 316.25, 317.5, 322.5, 323.75,
      403.75, 405., 410., 411.25, 416.25, 417.5, 422.5, 423.75,
      503.75, 505., 510., 511.25, 516.25, 517.5, 522.5, 523.75};
  const std::vector<DataType> rois = {0., 0., 0.5, 0.5};
  const std::vector<int64_t> batch_indices = {1};
  roi_align::RoiAlignParams params;
  params.batch = 3;
  params.channels = 3;
  params.in_height = 10;
  params.in_width = 10;
  params.out_height = 4;
  params.out_width = 2;
  params.num_rois = 1;
  params.sampling_ratio = 0;
  params.spatial_scale = 5.0;
  this->test_roi_align(rois, batch_indices, exp_out_max_pool, exp_out_avg_pool,
                       params);
}
