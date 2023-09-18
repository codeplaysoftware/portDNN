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

#include "portdnn/padding_mode.h"

#include "portdnn/pooling/operators.h"

#include "test/pooling/pooling_fixture.h"
#include "test/types/cartesian_product.h"
#include "test/types/data_format_types.h"
#include "test/types/kernel_data_types.h"
#include "test/types/nested_pairs_to_triple.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"

#include <array>
#include <vector>

using DataTypeList = sycldnn::types::KernelDataTypes;
using GTestTypeList = sycldnn::types::ToGTestTypes<DataTypeList>::type;
using DataFormatList = sycldnn::types::DataFormatTypes;
using BackendList = sycldnn::types::DefaultBackendTypes;

using SNNTypePairs =
    sycldnn::types::CartesianProduct<DataTypeList, DataFormatList>::type;
using SNNTypeBackendPairs =
    sycldnn::types::CartesianProduct<SNNTypePairs, BackendList>::type;
using TestTriples =
    sycldnn::types::NestedPairsToTriple<SNNTypeBackendPairs>::type;
using GTestTypeTriples = sycldnn::types::ToGTestTypes<TestTriples>::type;

template <typename Triple>
using PoolingOffsetAvgForward =
    PoolingFixture<typename Triple::FirstType, typename Triple::SecondType,
                   typename Triple::ThirdType, sycldnn::pooling::Average,
                   sycldnn::pooling::Forward>;

template <typename Triple>
using PoolingOffsetMaxForward =
    PoolingFixture<typename Triple::FirstType, typename Triple::SecondType,
                   typename Triple::ThirdType, sycldnn::pooling::Max,
                   sycldnn::pooling::Forward>;

template <typename Triple>
using PoolingOffsetAvgBackprop =
    PoolingFixture<typename Triple::FirstType, typename Triple::SecondType,
                   typename Triple::ThirdType, sycldnn::pooling::Average,
                   sycldnn::pooling::Backpropagate>;

template <typename Triple>
using PoolingOffsetMaxBackprop =
    PoolingFixture<typename Triple::FirstType, typename Triple::SecondType,
                   typename Triple::ThirdType, sycldnn::pooling::Max,
                   sycldnn::pooling::Backpropagate>;

TYPED_TEST_SUITE(PoolingOffsetAvgForward, GTestTypeTriples);
TYPED_TEST(PoolingOffsetAvgForward, Valid) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {181., 182., 183., 184., 197., 198.,
                                         199., 200., 405., 406., 407., 408.,
                                         421., 422., 423., 424.};
  const std::array<int, 4> in_shape = {{1, 11, 14, 4}};
  const auto padding = sycldnn::PaddingMode::VALID;
  const auto params = getPoolingParams<7, 4>(in_shape, padding);
  const DataType max_input_val = 2048.0;
  this->test_pool(exp_out, params, max_input_val, 0, 268);
}
TYPED_TEST(PoolingOffsetAvgForward, Same) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {
      131.,  132.,  133.,  134.,  145.,  146.,  147.,  148.,  157.,  158.,
      159.,  160.,  299.,  300.,  301.,  302.,  313.,  314.,  315.,  316.,
      325.,  326.,  327.,  328.,  443.,  444.,  445.,  446.,  457.,  458.,
      459.,  460.,  469.,  470.,  471.,  472.,  707.,  708.,  709.,  710.,
      721.,  722.,  723.,  724.,  733.,  734.,  735.,  736.,  875.,  876.,
      877.,  878.,  889.,  890.,  891.,  892.,  901.,  902.,  903.,  904.,
      1019., 1020., 1021., 1022., 1033., 1034., 1035., 1036., 1045., 1046.,
      1047., 1048., 1283., 1284., 1285., 1286., 1297., 1298., 1299., 1300.,
      1309., 1310., 1311., 1312., 1451., 1452., 1453., 1454., 1465., 1466.,
      1467., 1468., 1477., 1478., 1479., 1480., 1595., 1596., 1597., 1598.,
      1609., 1610., 1611., 1612., 1621., 1622., 1623., 1624.};
  const std::array<int, 4> in_shape = {{3, 12, 12, 4}};
  const auto padding = sycldnn::PaddingMode::SAME;
  const auto params = getPoolingParams<7, 4>(in_shape, padding);
  const DataType max_input_val = 2048.0;
  this->test_pool(exp_out, params, max_input_val, 338, 0);
}

TYPED_TEST_SUITE(PoolingOffsetMaxForward, GTestTypeTriples);
TYPED_TEST(PoolingOffsetMaxForward, Valid) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp = {11., 12., 15., 16.};
  const std::array<int, 4> in_shape = {1, 4, 4, 1};
  const auto padding = sycldnn::PaddingMode::VALID;
  const auto pp = getPoolingParams<3, 1>(in_shape, padding);
  this->test_pool(exp, pp, 2048, 16, 4);
}
TYPED_TEST(PoolingOffsetMaxForward, Same) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {
      237., 238., 239., 240., 253., 254., 255., 256., 261., 262., 263., 264.,
      413., 414., 415., 416., 429., 430., 431., 432., 437., 438., 439., 440.,
      501., 502., 503., 504., 517., 518., 519., 520., 525., 526., 527., 528.};
  const std::array<int, 4> in_shape = {{1, 12, 11, 4}};
  const auto padding = sycldnn::PaddingMode::SAME;
  const auto params = getPoolingParams<7, 4>(in_shape, padding);
  const DataType max_input_val = 2048.0;
  this->test_pool(exp_out, params, max_input_val, 2048, 2048);
}

TYPED_TEST_SUITE(PoolingOffsetAvgBackprop, GTestTypeTriples);
TYPED_TEST(PoolingOffsetAvgBackprop, Valid) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {
      0.04, 0.04, 0.12, 0.12, 0.12, 0.08, 0.08, 0.04, 0.04, 0.12,
      0.12, 0.12, 0.08, 0.08, 0.16, 0.16, 0.4,  0.4,  0.4,  0.24,
      0.24, 0.16, 0.16, 0.4,  0.4,  0.4,  0.24, 0.24, 0.16, 0.16,
      0.4,  0.4,  0.4,  0.24, 0.24, 0.12, 0.12, 0.28, 0.28, 0.28,
      0.16, 0.16, 0.12, 0.12, 0.28, 0.28, 0.28, 0.16, 0.16};
  const std::array<int, 4> in_shape = {{1, 7, 7, 1}};
  const auto padding = sycldnn::PaddingMode::VALID;
  const auto params = getPoolingParams<5, 2>(in_shape, padding);
  const DataType max_input_val = 2048.0;
  this->test_pool(exp_out, params, max_input_val, 64, 32);
}

TYPED_TEST(PoolingOffsetAvgBackprop, Same) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {
      5.196111111111111,  6.862777777777778,  8.686111111111112,
      8.671666666666667,  9.650833333333333,  11.963055555555556,
      10.296388888888888, 8.473055555555556,  9.242777777777778,
      12.029444444444444, 15.012777777777778, 14.531666666666666,
      15.760833333333334, 19.12638888888889,  16.33972222222222,
      13.356388888888889, 15.867777777777778, 20.454444444444444,
      25.28777777777778,  23.956666666666667, 25.498333333333335,
      30.447222222222223, 25.860555555555553, 21.02722222222222,
      26.312222222222225, 33.565555555555555, 41.065555555555555,
      37.95666666666666,  39.49833333333333,  46.224999999999994,
      38.971666666666664, 31.471666666666668, 24.38722222222222,
      31.040555555555553, 37.89055555555556,  34.83166666666666,
      36.06083333333333,  42.00416666666666,  35.350833333333334,
      28.500833333333333, 21.59388888888889,  27.447222222222226,
      33.45722222222223,  30.651666666666667, 31.630833333333335,
      36.73416666666667,  30.88083333333333,  24.870833333333334};
  const std::array<int, 4> in_shape = {{1, 6, 8, 1}};
  const auto padding = sycldnn::PaddingMode::SAME;
  const auto params = getPoolingParams<5, 1>(in_shape, padding);
  const DataType max_input_val = 2048.0;
  this->test_pool(exp_out, params, max_input_val, 128, 42);
}

TYPED_TEST_SUITE(PoolingOffsetMaxBackprop, GTestTypeTriples);
TYPED_TEST(PoolingOffsetMaxBackprop, Valid) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {
      1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,  10., 11., 12.,
      13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.,
      25., 26., 27., 28., 29., 30., 31., 32., 33., 34., 35., 36.,
      37., 38., 39., 40., 41., 42., 43., 44., 45., 46., 47., 48.};
  const std::array<int, 4> in_shape = {{3, 2, 2, 4}};
  const auto padding = sycldnn::PaddingMode::VALID;
  const auto params = getPoolingParams<1, 1>(in_shape, padding);
  const DataType max_input_val = 2048.0;
  this->test_pool(exp_out, params, max_input_val, 4, 65536);
}
TYPED_TEST(PoolingOffsetMaxBackprop, Same) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp_out = {
      0.,    0.,    0.,    0.,     0.,    0.,    0.,    0.,    0.,    0.,
      0.,    0.,    0.,    0.,     0.,    0.,    0.,    0.,    0.,    0.,
      0.,    0.,    0.,    0.,     0.,    0.,    0.,    0.,    0.,    0.,
      0.,    0.,    0.,    0.,     0.,    0.,    0.,    0.,    0.,    0.,
      0.,    0.,    0.,    0.,     0.,    0.,    0.,    0.,    0.,    0.,
      0.,    0.,    0.,    0.,     0.,    0.,    0.,    0.,    0.,    0.,
      0.,    0.,    0.,    0.,     0.,    0.,    0.,    0.,    0.,    0.,
      1.,    2.,    3.,    4.,     5.,    6.,    7.,    63.,   0.,    0.,
      0.,    0.,    0.,    14.,    15.,   16.,   17.,   18.,   19.,   20.,
      141.,  0.,    0.,    0.,     0.,    0.,    27.,   28.,   29.,   30.,
      31.,   32.,   33.,   219.,   0.,    0.,    0.,    0.,    0.,    40.,
      41.,   42.,   43.,   44.,    45.,   46.,   297.,  0.,    0.,    0.,
      0.,    0.,    53.,   54.,    55.,   56.,   57.,   58.,   59.,   375.,
      0.,    0.,    0.,    0.,     0.,    66.,   67.,   68.,   69.,   70.,
      71.,   72.,   453.,  0.,     0.,    0.,    0.,    0.,    79.,   80.,
      81.,   82.,   83.,   84.,    85.,   531.,  0.,    0.,    0.,    0.,
      0.,    92.,   93.,   94.,    95.,   96.,   97.,   98.,   609.,  0.,
      0.,    0.,    0.,    0.,     825.,  831.,  837.,  843.,  849.,  855.,
      861.,  5292., 0.,    0.,     0.,    0.,    0.,    0.,    0.,    0.,
      0.,    0.,    0.,    0.,     0.,    0.,    0.,    0.,    0.,    0.,
      0.,    0.,    0.,    0.,     0.,    0.,    0.,    0.,    0.,    0.,
      0.,    0.,    0.,    0.,     0.,    0.,    0.,    0.,    0.,    0.,
      0.,    0.,    0.,    0.,     0.,    0.,    0.,    0.,    0.,    0.,
      0.,    0.,    0.,    0.,     0.,    0.,    0.,    0.,    0.,    0.,
      0.,    0.,    0.,    0.,     0.,    0.,    0.,    0.,    0.,    0.,
      0.,    0.,    183.,  184.,   185.,  186.,  187.,  188.,  189.,  1155.,
      0.,    0.,    0.,    0.,     0.,    196.,  197.,  198.,  199.,  200.,
      201.,  202.,  1233., 0.,     0.,    0.,    0.,    0.,    209.,  210.,
      211.,  212.,  213.,  214.,   215.,  1311., 0.,    0.,    0.,    0.,
      0.,    222.,  223.,  224.,   225.,  226.,  227.,  228.,  1389., 0.,
      0.,    0.,    0.,    0.,     235.,  236.,  237.,  238.,  239.,  240.,
      241.,  1467., 0.,    0.,     0.,    0.,    0.,    248.,  249.,  250.,
      251.,  252.,  253.,  254.,   1545., 0.,    0.,    0.,    0.,    0.,
      261.,  262.,  263.,  264.,   265.,  266.,  267.,  1623., 0.,    0.,
      0.,    0.,    0.,    274.,   275.,  276.,  277.,  278.,  279.,  280.,
      1701., 0.,    0.,    0.,     0.,    0.,    1917., 1923., 1929., 1935.,
      1941., 1947., 1953., 11844., 0.,    0.,    0.,    0.,    0.,    0.,
      0.,    0.,    0.,    0.,     0.,    0.,    0.,    0.,    0.,    0.,
      0.,    0.,    0.,    0.,     0.,    0.,    0.,    0.,    0.,    0.,
      0.,    0.,    0.,    0.,     0.,    0.,    0.,    0.,    0.,    0.,
      0.,    0.,    0.,    0.,     0.,    0.,    0.,    0.,    0.,    0.,
      0.,    0.,    0.,    0.,     0.,    0.,    0.,    0.,    0.,    0.,
      0.,    0.,    0.,    0.,     0.,    0.,    0.,    0.,    0.,    0.,
      0.,    0.,    0.,    0.,     365.,  366.,  367.,  368.,  369.,  370.,
      371.,  2247., 0.,    0.,     0.,    0.,    0.,    378.,  379.,  380.,
      381.,  382.,  383.,  384.,   2325., 0.,    0.,    0.,    0.,    0.,
      391.,  392.,  393.,  394.,   395.,  396.,  397.,  2403., 0.,    0.,
      0.,    0.,    0.,    404.,   405.,  406.,  407.,  408.,  409.,  410.,
      2481., 0.,    0.,    0.,     0.,    0.,    417.,  418.,  419.,  420.,
      421.,  422.,  423.,  2559.,  0.,    0.,    0.,    0.,    0.,    430.,
      431.,  432.,  433.,  434.,   435.,  436.,  2637., 0.,    0.,    0.,
      0.,    0.,    443.,  444.,   445.,  446.,  447.,  448.,  449.,  2715.,
      0.,    0.,    0.,    0.,     0.,    456.,  457.,  458.,  459.,  460.,
      461.,  462.,  2793., 0.,     0.,    0.,    0.,    0.,    3009., 3015.,
      3021., 3027., 3033., 3039.,  3045., 18396.};
  const std::array<int, 4> in_shape = {{3, 14, 13, 1}};
  const auto padding = sycldnn::PaddingMode::SAME;
  const auto params = getPoolingParams<11, 1>(in_shape, padding);
  const DataType max_input_val = 2048.0;
  this->test_pool(exp_out, params, max_input_val, 32, 32, 64, 64);
}
