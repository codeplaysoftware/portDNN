/*
 * Copyright 2019 Codeplay Software Ltd.
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
#include <vector>

#include "sycldnn/padding_mode.h"
#include "sycldnn/status.h"

#include "sycldnn/backend/snn_backend.h"

#include "sycldnn/conv2d/conv_type.h"
#include "sycldnn/conv2d/launch.h"
#include "sycldnn/conv2d/params.h"
#include "sycldnn/conv2d/sizes.h"
#include "sycldnn/conv2d/workspace_size.h"

#include "sycldnn/conv2d/selector/direct_selector.h"
#include "sycldnn/conv2d/selector/im2col_selector.h"
#include "sycldnn/conv2d/selector/winograd_selector.h"

#include "sycldnn/helpers/padding.h"
#include "sycldnn/helpers/scope_exit.h"

#include "test/backend/backend_test_fixture.h"
#include "test/gen/iota_initialised_data.h"
#include "test/helpers/float_comparison.h"

#include "test/types/cartesian_product.h"
#include "test/types/kernel_data_types.h"
#include "test/types/nested_pairs_to_triple.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"
#include "test/types/type_list.h"

namespace {

template <typename Triple>
struct WorkspaceComparativeConv2D
    : public BackendTestFixture<sycldnn::backend::SNNBackend> {
  using SelectorType = typename Triple::FirstType;
  using DataType = typename Triple::SecondType;
  using Backend = sycldnn::backend::SNNBackend;
  using ConvType = typename Triple::ThirdType;

 protected:
  /**
   * Compare the output to the convolution using the Direct reference
   * implementation to the output given when using the implementation specified
   * by the template parameter SelectorType.
   *
   * Uses the standard test setup of using iota initialized data for the inputs
   * to the convolutions.
   *
   * \param params Convolution parameters to test.
   * \param use_recommended_size Whether to use the recommended size of
   * workspace, or only the required size.
   * \param max_val The maximum value to use in the input tensors, as used by
   * the iota_initialised_data function.
   */
  void test_conv(sycldnn::conv2d::Conv2DParams const& params,
                 bool use_recommended_size,
                 DataType max_val = static_cast<DataType>(2048)) {
    auto conv_sizes = sycldnn::conv2d::get_sizes<ConvType>(params);

    std::vector<DataType> input =
        iota_initialised_data(conv_sizes.input_size, max_val);
    std::for_each(begin(input), end(input), [](DataType& val) { val /= 1000; });
    std::vector<DataType> filter =
        iota_initialised_data(conv_sizes.filter_size, max_val);
    std::for_each(begin(filter), end(filter),
                  [](DataType& val) { val /= 1000; });
    std::vector<DataType> exp_output(conv_sizes.output_size,
                                     static_cast<DataType>(0));
    std::vector<DataType> output(conv_sizes.output_size,
                                 static_cast<DataType>(0));

    auto& provider = this->provider_;
    auto& backend = provider.get_backend();

    auto inp_gpu =
        provider.get_initialised_device_memory(conv_sizes.input_size, input);
    auto fil_gpu =
        provider.get_initialised_device_memory(conv_sizes.filter_size, filter);
    auto exp_out_gpu = provider.get_initialised_device_memory(
        conv_sizes.output_size, exp_output);
    auto out_gpu =
        provider.get_initialised_device_memory(conv_sizes.output_size, output);
    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(inp_gpu);
      provider.deallocate_ptr(fil_gpu);
      provider.deallocate_ptr(exp_out_gpu);
      provider.deallocate_ptr(out_gpu);
    };

    sycldnn::conv2d::DirectSelector direct_selector{};
    try {
      auto status = sycldnn::conv2d::launch<DataType, ConvType>(
          inp_gpu, fil_gpu, exp_out_gpu, params, direct_selector, backend);

      ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
      status.event.wait_and_throw();
    } catch (cl::sycl::exception const& e) {
      throw std::runtime_error(e.what());
    }
    provider.copy_device_data_to_host(conv_sizes.output_size, exp_out_gpu,
                                      exp_output);

    SelectorType selector{};
    ASSERT_NE(selector.select(params),
              sycldnn::conv2d::Algorithm::NotSupported);
    try {
      auto workspace_size_struct =
          sycldnn::conv2d::query_workspace_size<ConvType>(params, selector);
      auto workspace_size = use_recommended_size
                                ? workspace_size_struct.recommended_size
                                : workspace_size_struct.required_size;
      std::vector<DataType> workspace_vals(workspace_size);
      auto workspace = provider.get_initialised_device_memory(workspace_size,
                                                              workspace_vals);
      SNN_ON_SCOPE_EXIT { provider.deallocate_ptr(workspace); };
      auto status = sycldnn::conv2d::launch<DataType, ConvType>(
          inp_gpu, fil_gpu, out_gpu, params, selector, backend, workspace,
          workspace_size);

      ASSERT_EQ(sycldnn::StatusCode::OK, status.status);
      status.event.wait_and_throw();
    } catch (cl::sycl::exception const& e) {
      throw std::runtime_error(e.what());
    }
    provider.copy_device_data_to_host(conv_sizes.output_size, out_gpu, output);

    for (size_t i = 0; i < exp_output.size(); ++i) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      // Allow a reasonably large tolerance of 512 ULPs, as we are comparing
      // two different algorithmic approaches, which will both have different
      // rounding patterns. The correctness of the algorithms is affirmed in
      // the other convolution tests.
      SNN_ALMOST_EQUAL(exp_output[i], output[i], 512u);
    }
  }
};

using DataTypeList = sycldnn::types::KernelDataTypes;
using SelectorList =
    sycldnn::types::TypeList<sycldnn::conv2d::Im2colSelector,
                             sycldnn::conv2d::WinogradSelector,
                             sycldnn::conv2d::WinogradLargeSelector>;
using ConvTypeList =
    sycldnn::types::TypeList<sycldnn::conv2d::conv_type::Forward,
                             sycldnn::conv2d::conv_type::InputBackprop,
                             sycldnn::conv2d::conv_type::FilterBackprop>;

using SNNTestPairs =
    sycldnn::types::CartesianProduct<SelectorList, DataTypeList>::type;
using TestPairsWithConvType =
    sycldnn::types::CartesianProduct<SNNTestPairs, ConvTypeList>::type;
using TestTriples =
    sycldnn::types::NestedPairsToTriple<TestPairsWithConvType>::type;

using GTestTypeTriples = sycldnn::types::ToGTestTypes<TestTriples>::type;
TYPED_TEST_CASE(WorkspaceComparativeConv2D, GTestTypeTriples);

/**
 * Rather than using the full sized VGG model, which would take longer to
 * compute each test than ideal, scale down each feature set by a fixed amount
 * to reduce the time spent in each test.
 */
int channel_scale(int x) { return x / 8; }

/**
 * Similarly we scale the image sizes down by a fixed amount to reduce test
 * time.
 */
int image_scale(int x) { return x / 4; }

sycldnn::conv2d::Conv2DParams vgg1_params() {
  sycldnn::conv2d::Conv2DParams params;
  params.channels = 3;
  params.features = channel_scale(64);
  params.batch = 4;
  params.in_rows = image_scale(224);
  params.in_cols = image_scale(224);
  params.window_rows = 3;
  params.window_cols = 3;
  params.stride_rows = 1;
  params.stride_cols = 1;
  return sycldnn::helpers::add_padding_to(params, sycldnn::PaddingMode::SAME);
}

sycldnn::conv2d::Conv2DParams vgg4_params() {
  sycldnn::conv2d::Conv2DParams params;
  params.channels = channel_scale(128);
  params.features = channel_scale(128);
  params.batch = 4;
  params.in_rows = image_scale(112);
  params.in_cols = image_scale(112);
  params.window_rows = 3;
  params.window_cols = 3;
  params.stride_rows = 1;
  params.stride_cols = 1;
  return sycldnn::helpers::add_padding_to(params, sycldnn::PaddingMode::SAME);
}

sycldnn::conv2d::Conv2DParams vgg6_params() {
  sycldnn::conv2d::Conv2DParams params;
  params.channels = channel_scale(256);
  params.features = channel_scale(256);
  params.batch = 4;
  params.in_rows = image_scale(56);
  params.in_cols = image_scale(56);
  params.window_rows = 3;
  params.window_cols = 3;
  params.stride_rows = 1;
  params.stride_cols = 1;
  return sycldnn::helpers::add_padding_to(params, sycldnn::PaddingMode::SAME);
}

sycldnn::conv2d::Conv2DParams vgg8_params() {
  sycldnn::conv2d::Conv2DParams params;
  params.channels = channel_scale(512);
  params.features = channel_scale(512);
  params.batch = 4;
  params.in_rows = image_scale(28);
  params.in_cols = image_scale(28);
  params.window_rows = 3;
  params.window_cols = 3;
  params.stride_rows = 1;
  params.stride_cols = 1;
  return sycldnn::helpers::add_padding_to(params, sycldnn::PaddingMode::SAME);
}

sycldnn::conv2d::Conv2DParams vgg9_params() {
  sycldnn::conv2d::Conv2DParams params;
  params.channels = channel_scale(512);
  params.features = channel_scale(512);
  params.batch = 4;
  params.in_rows = image_scale(14);
  params.in_cols = image_scale(14);
  params.window_rows = 3;
  params.window_cols = 3;
  params.stride_rows = 1;
  params.stride_cols = 1;
  return sycldnn::helpers::add_padding_to(params, sycldnn::PaddingMode::SAME);
}

}  // namespace

TYPED_TEST(WorkspaceComparativeConv2D, Vgg1Required) {
  this->test_conv(vgg1_params(), false);
}

TYPED_TEST(WorkspaceComparativeConv2D, Vgg4Required) {
  this->test_conv(vgg4_params(), false);
}

TYPED_TEST(WorkspaceComparativeConv2D, Vgg6Required) {
  this->test_conv(vgg6_params(), false);
}

TYPED_TEST(WorkspaceComparativeConv2D, Vgg8Required) {
  this->test_conv(vgg8_params(), false);
}

TYPED_TEST(WorkspaceComparativeConv2D, Vgg9Required) {
  this->test_conv(vgg9_params(), false);
}

TYPED_TEST(WorkspaceComparativeConv2D, Vgg1Recommended) {
  this->test_conv(vgg1_params(), true);
}

TYPED_TEST(WorkspaceComparativeConv2D, Vgg4Recommended) {
  this->test_conv(vgg4_params(), true);
}

TYPED_TEST(WorkspaceComparativeConv2D, Vgg6Recommended) {
  this->test_conv(vgg6_params(), true);
}

TYPED_TEST(WorkspaceComparativeConv2D, Vgg8Recommended) {
  this->test_conv(vgg8_params(), true);
}

TYPED_TEST(WorkspaceComparativeConv2D, Vgg9Recommended) {
  this->test_conv(vgg9_params(), true);
}
