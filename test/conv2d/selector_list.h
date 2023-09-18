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
#ifndef PORTDNN_TEST_CONV2D_SELECTOR_LIST_H_
#define PORTDNN_TEST_CONV2D_SELECTOR_LIST_H_

#include "test/types/type_list.h"

#include "portdnn/conv2d/selector/direct_selector.h"
#include "portdnn/conv2d/selector/im2col_selector.h"
#include "portdnn/conv2d/selector/matmul_selector.h"
#include "portdnn/conv2d/selector/tiled_selector.h"
#include "portdnn/conv2d/selector/winograd_selector.h"

namespace sycldnn {
namespace types {

using SelectorList = sycldnn::types::TypeList<
    sycldnn::conv2d::DirectSelector, sycldnn::conv2d::TiledSelector,
    sycldnn::conv2d::Im2colSelector, sycldnn::conv2d::WinogradSelector,
    sycldnn::conv2d::MatmulSelector>;

}  // namespace types
}  // namespace sycldnn

#endif  // PORTDNN_TEST_CONV2D_SELECTOR_LIST_H_
