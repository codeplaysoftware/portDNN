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
#ifndef PORTDNN_TEST_TYPES_KERNEL_DATA_TYPES_H_
#define PORTDNN_TEST_TYPES_KERNEL_DATA_TYPES_H_

#include "test/types/to_gtest_types.h"
#include "test/types/type_list.h"

#ifdef SNN_USE_HALF
#include <CL/sycl.hpp>
#endif  // SNN_USE_HALF

namespace sycldnn {
namespace types {

#ifdef SNN_USE_DOUBLE
#define SNN_DOUBLE_LIST , double
#else
#define SNN_DOUBLE_LIST
#endif  // SNN_USE_DOUBLE

#ifdef SNN_USE_HALF
#define SNN_HALF_LIST , cl::sycl::half
#else
#define SNN_HALF_LIST
#endif  // SNN_USE_HALF

/**
 * List of data types to use in kernels.
 *
 * This list depends on which types are enabled at compile time.
 */
using KernelDataTypes = TypeList<float SNN_DOUBLE_LIST SNN_HALF_LIST>;
/** The same as KernelDataTypes but in the googletest Types format. */
using GTestKernelDataTypes = ToGTestTypes<KernelDataTypes>::type;

#undef SNN_DOUBLE_LIST
#undef SNN_HALF_LIST

}  // namespace types
}  // namespace sycldnn
#endif  // PORTDNN_TEST_TYPES_KERNEL_DATA_TYPES_H_
