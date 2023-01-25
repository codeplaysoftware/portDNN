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
#include <vector>

#include "test/matmul/fixture.h"
#include "test/types/cartesian_product.h"
#include "test/types/kernel_data_types.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"

using DataTypeList = sycldnn::types::KernelDataTypes;
using BackendTypeList = sycldnn::types::DefaultBackendTypes;
using TypePairList =
    sycldnn::types::CartesianProduct<DataTypeList, BackendTypeList>::type;
using GTestTypeList = sycldnn::types::ToGTestTypes<TypePairList>::type;

template <typename DataType>
using MatmulOffset = MatmulFixture<DataType, false, false>;
TYPED_TEST_SUITE(MatmulOffset, GTestTypeList);

TYPED_TEST(MatmulOffset, M4xK4xN4) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp = {
      1.,    2.,    3.,    4.,    5.,    6.,    7.,    8.,
      9.,    10.,   11.,   12.,   13.,   14.,   15.,   16.,
      1130., 1204., 1278., 1352., 1370., 1460., 1550., 1640.,
      1610., 1716., 1822., 1928., 1850., 1972., 2094., 2216.};
  const int batches = 1;
  const int m = 4;
  const int k = 4;
  const int n = 4;
  const auto beta = static_cast<DataType>(0);
  const int lhs_offset = 16;
  const int rhs_offset = 8;
  const int out_offset = 16;

  this->run(exp, batches, m, k, n, beta, lhs_offset, rhs_offset, out_offset, 0);
}
TYPED_TEST(MatmulOffset, M4xK2xN4) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp = {
      1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   123., 134.,
      145., 156., 167., 182., 197., 212., 211., 230., 249., 268.,
      255., 278., 301., 324., 515., 542., 569., 596., 591., 622.,
      653., 684., 667., 702., 737., 772., 743., 782., 821., 860.};
  const int batches = 2;
  const int m = 4;
  const int k = 2;
  const int n = 4;
  const auto beta = static_cast<DataType>(0);
  const int lhs_offset = 4;
  const int rhs_offset = 8;
  const int out_offset = 8;

  this->run(exp, batches, m, k, n, beta, lhs_offset, rhs_offset, out_offset, 0);
}
