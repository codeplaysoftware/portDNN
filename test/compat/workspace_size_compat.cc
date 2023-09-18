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

static void snnParamsToDesc(TensorDescriptor& xDesc, TensorDescriptor& yDesc,
                            FilterDescriptor& wDesc,
                            ConvolutionDescriptor& convDesc,
                            const sycldnn::conv2d::Conv2DParams& conv_params) {
  setTensor4dDescriptor(xDesc, sycldnn::DataFormat::NCHW,
                        SNNDataType::SNN_FLOAT, conv_params.batch,
                        conv_params.channels, conv_params.in_rows,
                        conv_params.in_cols);

  setTensor4dDescriptor(yDesc, sycldnn::DataFormat::NCHW,
                        SNNDataType::SNN_FLOAT, conv_params.batch,
                        conv_params.channels, conv_params.out_rows,
                        conv_params.out_cols);

  setFilter4dDescriptor(wDesc, SNNDataType::SNN_FLOAT,
                        sycldnn::DataFormat::NCHW, conv_params.features,
                        conv_params.channels, conv_params.window_rows,
                        conv_params.window_cols);

  convDesc.set2d(conv_params.pad_rows, conv_params.pad_cols,
                 conv_params.stride_rows, conv_params.stride_cols,
                 conv_params.dilation_rows, conv_params.dilation_cols);
}

// TODO: Extend the Testing for the other algorithms after the wrapper
// integrates them
TEST(Conv2DWorskpaceSize, DirectNoWorkspace) {
  SNNHandle handle;
  auto params = get_params(3, 1, 56, 256, 256, 1, sycldnn::PaddingMode::SAME);
  TensorDescriptor xDesc, yDesc;
  FilterDescriptor wDesc;
  ConvolutionDescriptor convDesc;
  snnParamsToDesc(xDesc, yDesc, wDesc, convDesc, params);
  size_t forward_workspace{};
  getConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc,
                                     conv2d::Algorithm::Direct,
                                     &forward_workspace);
  EXPECT_EQ(0u, forward_workspace);

  size_t inbk_workspace{};
  getConvolutionBackwardDataWorkspaceSize(handle, wDesc, yDesc, convDesc, xDesc,
                                          conv2d::Algorithm::Direct,
                                          &inbk_workspace);
  EXPECT_EQ(0u, inbk_workspace);

  size_t filbk_workspace{};
  getConvolutionBackwardFilterWorkspaceSize(handle, xDesc, yDesc, convDesc,
                                            wDesc, conv2d::Algorithm::Direct,
                                            &filbk_workspace);
  EXPECT_EQ(0u, filbk_workspace);
}

TEST(Conv2DWorskpaceSize, TiledNoWorkspace) {
  SNNHandle handle;
  SNNCreate(handle);
  auto params = get_params(3, 1, 56, 256, 256, 1, sycldnn::PaddingMode::SAME);
  TensorDescriptor xDesc, yDesc;
  FilterDescriptor wDesc;
  ConvolutionDescriptor convDesc;
  snnParamsToDesc(xDesc, yDesc, wDesc, convDesc, params);
  size_t forward_workspace{};
  getConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc,
                                     conv2d::Algorithm::Tiled,
                                     &forward_workspace);
  EXPECT_EQ(0u, forward_workspace);

  size_t inbk_workspace{};
  getConvolutionBackwardDataWorkspaceSize(handle, wDesc, yDesc, convDesc, xDesc,
                                          conv2d::Algorithm::Tiled,
                                          &inbk_workspace);
  EXPECT_EQ(0u, inbk_workspace);

  size_t filbk_workspace{};
  getConvolutionBackwardFilterWorkspaceSize(handle, xDesc, yDesc, convDesc,
                                            wDesc, conv2d::Algorithm::Tiled,
                                            &filbk_workspace);
  EXPECT_EQ(0u, filbk_workspace);
}

TEST(Conv2DWorskpaceSize, Im2colVGGLayer1Workspace) {
  // We allow the queried workspace to be larger than the absolute minimum
  // required, so that internally we can add extra size requirements for
  // paddding or alignment.
  SNNHandle handle;
  SNNCreate(handle);
  auto params = get_params(3, 1, 224, 64, 64, 32, sycldnn::PaddingMode::SAME);
  TensorDescriptor xDesc, yDesc;
  FilterDescriptor wDesc;
  ConvolutionDescriptor convDesc;
  snnParamsToDesc(xDesc, yDesc, wDesc, convDesc, params);
  size_t forward_workspace{};
  getConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc,
                                     conv2d::Algorithm::Im2col,
                                     &forward_workspace);
  auto constexpr fwd_n_tiles = 224u * 224u;
  auto constexpr fwd_tile_size = 3u * 3u * 64u;
  EXPECT_LE(32u * fwd_n_tiles * fwd_tile_size, forward_workspace);

  size_t inbk_workspace{};
  getConvolutionBackwardDataWorkspaceSize(handle, wDesc, yDesc, convDesc, xDesc,
                                          conv2d::Algorithm::Im2col,
                                          &inbk_workspace);
  auto constexpr inbk_n_tiles = 224u * 224u;
  auto constexpr inbk_tile_size = 3u * 3u * 64u;
  auto constexpr inbk_fil_size = 3u * 3u * 64u * 64;
  EXPECT_LE(32u * inbk_n_tiles * inbk_tile_size + inbk_fil_size,
            inbk_workspace);

  size_t filbk_workspace{};
  getConvolutionBackwardFilterWorkspaceSize(handle, xDesc, yDesc, convDesc,
                                            wDesc, conv2d::Algorithm::Im2col,
                                            &filbk_workspace);
  auto constexpr filbk_n_tiles = 3u * 3u * 64u;
  auto constexpr filbk_tile_size = 224u * 224u;
  EXPECT_LE(32u * filbk_n_tiles * filbk_tile_size, filbk_workspace);
}
