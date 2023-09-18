/*
 * Copyright Codeplay Software Ltd.
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

#include "portdnn/compat/convolution.hpp"
#include "portdnn/helpers/padding.h"
#include "portdnn/padding_mode.h"
#include "test/gen/iota_initialised_data.h"

#include <type_traits>

using namespace sycldnn;
using namespace sycldnn::compat;

sycldnn::conv2d::Conv2DParams get_params(int window, int stride, int size,
                                         int channels, int features, int batch,
                                         sycldnn::PaddingMode padding) {
  sycldnn::conv2d::Conv2DParams params;
  params.window_rows = window;
  params.window_cols = window;
  params.stride_rows = stride;
  params.stride_cols = stride;
  params.in_rows = size;
  params.in_cols = size;
  params.channels = channels;
  params.features = features;
  params.batch = batch;
  params.dilation_rows = 1;
  params.dilation_cols = 1;

  return sycldnn::helpers::add_padding_to(params, padding);
}

static void initDescriptors(TensorDescriptor& inDesc, FilterDescriptor& fDesc,
                            ConvolutionDescriptor& convDesc,
                            TensorDescriptor& out_desc) {
  // input
  const int in_n = 1;
  const int in_c = 1;
  const int in_h = 5;
  const int in_w = 5;

  inDesc.set4d(sycldnn::DataFormat::NCHW, in_n, in_c, in_h, in_w);

  // filter
  const int filt_k = 1;
  const int filt_c = 1;
  const int filt_h = 2;
  const int filt_w = 2;

  fDesc.set4d(sycldnn::DataFormat::NCHW, filt_k, filt_c, filt_h, filt_w);

  // convolution
  const int pad_h = 1;
  const int pad_w = 1;
  const int str_h = 1;
  const int str_w = 1;
  const int dil_h = 1;
  const int dil_w = 1;

  convDesc.set2d(pad_h, pad_w, str_h, str_w, dil_h, dil_w);

  // output
  int out_n;
  int out_c;
  int out_h;
  int out_w;

  getConvolution2dForwardOutputDim(convDesc, inDesc, fDesc, &out_n, &out_c,
                                   &out_h, &out_w);

  out_desc.set4d(sycldnn::DataFormat::NCHW, out_n, out_c, out_h, out_w);
}
// TODO: Extend the Testing for the other algorithms after the wrapper
// integrates them
TEST(Conv2DFindAlgorithm, Forward) {
  SNNHandle handle;
  SNNCreate(handle);
  TensorDescriptor in_desc;
  FilterDescriptor filt_desc(4);
  ConvolutionDescriptor conv_desc;
  TensorDescriptor out_desc;
  initDescriptors(in_desc, filt_desc, conv_desc, out_desc);

  int returnedAlgoCount{};
  convolutionFwdAlgoPerf_t results{};
  auto status = findConvolutionForwardAlgorithm(handle, in_desc, filt_desc,
                                                conv_desc, out_desc, 1,
                                                &returnedAlgoCount, &results);
  EXPECT_EQ(status.status, StatusCode::OK);
  EXPECT_EQ(results.status[0].status, StatusCode::OK);
  EXPECT_GT(results.time[0], 0);
  EXPECT_NE(results.algo[0], sycldnn::conv2d::Algorithm::NotSupported);
}

TEST(Conv2DFindAlgorithm, InputBackprop) {
  SNNHandle handle;
  SNNCreate(handle);

  TensorDescriptor in_desc;
  FilterDescriptor filt_desc(4);
  ConvolutionDescriptor conv_desc;
  TensorDescriptor out_desc;
  initDescriptors(in_desc, filt_desc, conv_desc, out_desc);

  int returnedAlgoCount{};
  convolutionFwdAlgoPerf_t results{};
  auto status = findConvolutionBackwardDataAlgorithm(
      handle, filt_desc, out_desc, conv_desc, in_desc, 1, &returnedAlgoCount,
      &results);
  EXPECT_EQ(status.status, StatusCode::OK);
  EXPECT_EQ(results.status[0].status, StatusCode::OK);
  EXPECT_GT(results.time[0], 0);
  EXPECT_NE(results.algo[0], sycldnn::conv2d::Algorithm::NotSupported);
}
