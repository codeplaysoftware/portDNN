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

#include "portdnn/helpers/padding.h"

#include "portdnn/conv2d/conv_type.h"
#include "portdnn/conv2d/params.h"
#include "portdnn/conv2d/workspace_size.h"

#include "portdnn/conv2d/selector/direct_selector.h"
#include "portdnn/conv2d/selector/im2col_selector.h"
#include "portdnn/conv2d/selector/tiled_selector.h"
#include "portdnn/conv2d/selector/winograd_selector.h"

#include <stddef.h>

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

TEST(Conv2DWorskpaceSize, DirectNoWorkspace) {
  sycldnn::conv2d::DirectSelector direct_selector{};
  auto params = get_params(3, 1, 56, 256, 256, 1, sycldnn::PaddingMode::SAME);

  auto forward_workspace = sycldnn::conv2d::query_workspace_size<
      sycldnn::conv2d::conv_type::Forward>(params, direct_selector);
  EXPECT_EQ(0u, forward_workspace.required_size);
  EXPECT_EQ(0u, forward_workspace.recommended_size);

  auto inbk_workspace = sycldnn::conv2d::query_workspace_size<
      sycldnn::conv2d::conv_type::InputBackprop>(params, direct_selector);
  EXPECT_EQ(0u, inbk_workspace.required_size);
  EXPECT_EQ(0u, inbk_workspace.recommended_size);

  auto filbk_workspace = sycldnn::conv2d::query_workspace_size<
      sycldnn::conv2d::conv_type::FilterBackprop>(params, direct_selector);
  EXPECT_EQ(0u, filbk_workspace.required_size);
  EXPECT_EQ(0u, filbk_workspace.recommended_size);
}

TEST(Conv2DWorskpaceSize, TiledNoWorkspace) {
  sycldnn::conv2d::TiledSelector selector{};
  auto params = get_params(3, 1, 56, 256, 256, 1, sycldnn::PaddingMode::SAME);

  auto forward_workspace = sycldnn::conv2d::query_workspace_size<
      sycldnn::conv2d::conv_type::Forward>(params, selector);
  EXPECT_EQ(0u, forward_workspace.required_size);
  EXPECT_EQ(0u, forward_workspace.recommended_size);

  auto inbk_workspace = sycldnn::conv2d::query_workspace_size<
      sycldnn::conv2d::conv_type::InputBackprop>(params, selector);
  EXPECT_EQ(0u, inbk_workspace.required_size);
  EXPECT_EQ(0u, inbk_workspace.recommended_size);

  auto filbk_workspace = sycldnn::conv2d::query_workspace_size<
      sycldnn::conv2d::conv_type::FilterBackprop>(params, selector);
  EXPECT_EQ(0u, filbk_workspace.required_size);
  EXPECT_EQ(0u, filbk_workspace.recommended_size);
}

TEST(Conv2DWorskpaceSize, Im2colVGGLayer1Workspace) {
  // We allow the queried workspace to be larger than the absolute minimum
  // required, so that internally we can add extra size requirements for
  // paddding or alignment.
  sycldnn::conv2d::Im2colSelector selector{};
  auto params = get_params(3, 1, 224, 64, 64, 32, sycldnn::PaddingMode::SAME);

  auto constexpr fwd_n_tiles = 224u * 224u;
  auto constexpr fwd_tile_size = 3u * 3u * 64u;
  auto forward_workspace = sycldnn::conv2d::query_workspace_size<
      sycldnn::conv2d::conv_type::Forward>(params, selector);
  EXPECT_LE(fwd_n_tiles * fwd_tile_size, forward_workspace.required_size);
  EXPECT_LE(32u * fwd_n_tiles * fwd_tile_size,
            forward_workspace.recommended_size);

  auto constexpr inbk_n_tiles = 224u * 224u;
  auto constexpr inbk_tile_size = 3u * 3u * 64u;
  auto constexpr inbk_fil_size = 3u * 3u * 64u * 64;
  auto inbk_workspace = sycldnn::conv2d::query_workspace_size<
      sycldnn::conv2d::conv_type::InputBackprop>(params, selector);
  EXPECT_LE(inbk_n_tiles * inbk_tile_size + inbk_fil_size,
            inbk_workspace.required_size);
  EXPECT_LE(32u * inbk_n_tiles * inbk_tile_size + inbk_fil_size,
            inbk_workspace.recommended_size);

  auto constexpr filbk_n_tiles = 3u * 3u * 64u;
  auto constexpr filbk_tile_size = 224u * 224u;
  auto filbk_workspace = sycldnn::conv2d::query_workspace_size<
      sycldnn::conv2d::conv_type::FilterBackprop>(params, selector);
  EXPECT_LE(filbk_n_tiles * filbk_tile_size, filbk_workspace.required_size);
  EXPECT_LE(32u * filbk_n_tiles * filbk_tile_size,
            filbk_workspace.recommended_size);
}

TEST(Conv2DWorskpaceSize, WinogradVGGLayer1Workspace) {
  // TODO(jwlawson): How do we handle different Winograd tile sizes?
  sycldnn::conv2d::WinogradSelector selector{};
  auto params = get_params(3, 1, 224, 64, 64, 32, sycldnn::PaddingMode::SAME);
  auto ceil = [](size_t a, size_t b) { return (a + b - 1) / b; };

  auto constexpr M = 2;
  auto constexpr N = 2;
  auto constexpr A = M + 3u - 1;
  auto constexpr B = N + 3u - 1;

  auto fwd_in_tiles = ceil(224u, M) * ceil(224u, N) * 64u;
  auto fwd_fil_tiles = 64u * 64u;
  auto fwd_out_tiles = ceil(224u, M) * ceil(224u, N) * 64u;
  auto forward_workspace = sycldnn::conv2d::query_workspace_size<
      sycldnn::conv2d::conv_type::Forward>(params, selector);
  EXPECT_LE(A * B * (fwd_in_tiles + fwd_fil_tiles + fwd_out_tiles),
            forward_workspace.required_size);
  EXPECT_LE(A * B * (32u * (fwd_in_tiles + fwd_out_tiles) + fwd_fil_tiles),
            forward_workspace.recommended_size);

  auto inbk_in_tiles = ceil(224u, M) * ceil(224u, N) * 64u;
  auto inbk_fil_tiles = 64u * 64u;
  auto inbk_out_tiles = ceil(224u, M) * ceil(224u, N) * 64u;
  auto inbk_workspace = sycldnn::conv2d::query_workspace_size<
      sycldnn::conv2d::conv_type::InputBackprop>(params, selector);
  EXPECT_LE(A * B * (inbk_in_tiles + inbk_fil_tiles + inbk_out_tiles),
            inbk_workspace.required_size);
  EXPECT_LE(A * B * (32u * (inbk_in_tiles + inbk_out_tiles) + inbk_fil_tiles),
            inbk_workspace.recommended_size);

  auto filbk_in_tiles = ceil(224u, M) * ceil(224u, N) * 64u;
  auto filbk_fil_tiles = ceil(224u, M) * ceil(224u, N) * 64u;
  auto filbk_out_tiles = 64u * 64u;
  auto filbk_workspace = sycldnn::conv2d::query_workspace_size<
      sycldnn::conv2d::conv_type::FilterBackprop>(params, selector);
  EXPECT_LE(A * B * (filbk_in_tiles + filbk_fil_tiles + filbk_out_tiles),
            filbk_workspace.required_size);
  EXPECT_LE(
      A * B * (32u * (filbk_in_tiles + filbk_out_tiles) + filbk_fil_tiles),
      filbk_workspace.recommended_size);
}
