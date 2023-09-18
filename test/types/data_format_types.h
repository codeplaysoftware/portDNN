/*
 * Copyright Codeplay Software Ltd
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
#ifndef PORTDNN_TEST_TYPES_DATA_FORMAT_TYPES_H_
#define PORTDNN_TEST_TYPES_DATA_FORMAT_TYPES_H_

#include "portdnn/format_type.h"

#include "test/types/type_list.h"

namespace sycldnn {
namespace types {

/**
 * List of data formats to use in kernels.
 *
 * We use intermediate types to represent data formats to workaround a GTest
 * limitation which does not allow for typed tests with parameters.
 */
using DataFormatTypes = TypeList<layout::NHWC, layout::NCHW>;

}  // namespace types
}  // namespace sycldnn
#endif  // PORTDNN_TEST_TYPES_DATA_FORMAT_TYPES_H_
