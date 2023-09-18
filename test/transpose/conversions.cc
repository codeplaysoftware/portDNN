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

#include <CL/sycl.hpp>

#include <stddef.h>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "portdnn/helpers/scope_exit.h"
#include "portdnn/status.h"
#include "portdnn/transpose/launch.h"

#include "test/backend/backend_test_fixture.h"
#include "test/gen/iota_initialised_data.h"

#include "test/types/cartesian_product.h"
#include "test/types/kernel_data_types.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"

using DataTypeList = sycldnn::types::KernelDataTypes;
using Backends = sycldnn::types::DefaultBackendTypes;

using TypeBackendPairs =
    sycldnn::types::CartesianProduct<DataTypeList, Backends>::type;

using GTestTypePairs = sycldnn::types::ToGTestTypes<TypeBackendPairs>::type;

template <typename Pair>
struct TransposeConversion
    : public BackendTestFixture<typename Pair::SecondType> {
 public:
  using DataType = typename Pair::FirstType;
};

TYPED_TEST_SUITE(TransposeConversion, GTestTypePairs);

TYPED_TEST(TransposeConversion, NHWCToNCHW) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp = {
      1.,  6.,  11., 16., 21., 26., 31., 36.,  41.,  46.,  51.,  56.,
      2.,  7.,  12., 17., 22., 27., 32., 37.,  42.,  47.,  52.,  57.,
      3.,  8.,  13., 18., 23., 28., 33., 38.,  43.,  48.,  53.,  58.,
      4.,  9.,  14., 19., 24., 29., 34., 39.,  44.,  49.,  54.,  59.,
      5.,  10., 15., 20., 25., 30., 35., 40.,  45.,  50.,  55.,  60.,
      61., 66., 71., 76., 81., 86., 91., 96.,  101., 106., 111., 116.,
      62., 67., 72., 77., 82., 87., 92., 97.,  102., 107., 112., 117.,
      63., 68., 73., 78., 83., 88., 93., 98.,  103., 108., 113., 118.,
      64., 69., 74., 79., 84., 89., 94., 99.,  104., 109., 114., 119.,
      65., 70., 75., 80., 85., 90., 95., 100., 105., 110., 115., 120.};
  const std::vector<int> sizes = {2, 3, 4, 5};
  const DataType max_val = 2048.0;

  size_t tensor_size = std::accumulate(begin(sizes), end(sizes), 1,
                                       [](int a, int b) { return a * b; });
  ASSERT_EQ(tensor_size, exp.size());

  std::vector<DataType> in_data = iota_initialised_data(tensor_size, max_val);
  std::vector<DataType> out_data = iota_initialised_data(tensor_size, max_val);

  auto& provider = this->provider_;
  auto& backend = provider.get_backend();

  {
    auto in_gpu = provider.get_initialised_device_memory(tensor_size, in_data);
    auto out_gpu =
        provider.get_initialised_device_memory(tensor_size, out_data);

    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(in_gpu);
      provider.deallocate_ptr(out_gpu);
    };

    try {
      auto status = sycldnn::transpose::convert_nhwc_to_nchw<DataType>(
          in_gpu, out_gpu, sizes, backend);

      ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
      status.event.wait_and_throw();
    } catch (cl::sycl::exception const& e) {
      throw std::runtime_error(e.what());
    }

    provider.copy_device_data_to_host(tensor_size, out_gpu, out_data);
  }

  for (size_t i = 0; i < exp.size(); ++i) {
    SCOPED_TRACE("Element: " + std::to_string(i));
    if (std::is_same<DataType, double>::value) {
      EXPECT_DOUBLE_EQ(exp[i], out_data[i]);
    } else {
      EXPECT_FLOAT_EQ(exp[i], out_data[i]);
    }
  }
}

TYPED_TEST(TransposeConversion, NCHWToNHWC) {
  using DataType = typename TestFixture::DataType;
  const std::vector<DataType> exp = {
      1.,  21., 41.,  2.,  22., 42.,  3.,  23., 43.,  4.,  24.,  44.,
      5.,  25., 45.,  6.,  26., 46.,  7.,  27., 47.,  8.,  28.,  48.,
      9.,  29., 49.,  10., 30., 50.,  11., 31., 51.,  12., 32.,  52.,
      13., 33., 53.,  14., 34., 54.,  15., 35., 55.,  16., 36.,  56.,
      17., 37., 57.,  18., 38., 58.,  19., 39., 59.,  20., 40.,  60.,
      61., 81., 101., 62., 82., 102., 63., 83., 103., 64., 84.,  104.,
      65., 85., 105., 66., 86., 106., 67., 87., 107., 68., 88.,  108.,
      69., 89., 109., 70., 90., 110., 71., 91., 111., 72., 92.,  112.,
      73., 93., 113., 74., 94., 114., 75., 95., 115., 76., 96.,  116.,
      77., 97., 117., 78., 98., 118., 79., 99., 119., 80., 100., 120.};

  const std::vector<int> sizes = {2, 3, 4, 5};
  const DataType max_val = 2048.0;

  size_t tensor_size = std::accumulate(begin(sizes), end(sizes), 1,
                                       [](int a, int b) { return a * b; });
  ASSERT_EQ(tensor_size, exp.size());

  std::vector<DataType> in_data = iota_initialised_data(tensor_size, max_val);
  std::vector<DataType> out_data = iota_initialised_data(tensor_size, max_val);

  auto& provider = this->provider_;
  auto& backend = provider.get_backend();

  {
    auto in_gpu = provider.get_initialised_device_memory(tensor_size, in_data);
    auto out_gpu =
        provider.get_initialised_device_memory(tensor_size, out_data);

    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(in_gpu);
      provider.deallocate_ptr(out_gpu);
    };

    try {
      auto status = sycldnn::transpose::convert_nchw_to_nhwc<DataType>(
          in_gpu, out_gpu, sizes, backend);

      ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
      status.event.wait_and_throw();
    } catch (cl::sycl::exception const& e) {
      throw std::runtime_error(e.what());
    }

    provider.copy_device_data_to_host(tensor_size, out_gpu, out_data);
  }

  for (size_t i = 0; i < exp.size(); ++i) {
    SCOPED_TRACE("Element: " + std::to_string(i));
    if (std::is_same<DataType, double>::value) {
      EXPECT_DOUBLE_EQ(exp[i], out_data[i]);
    } else {
      EXPECT_FLOAT_EQ(exp[i], out_data[i]);
    }
  }
}
