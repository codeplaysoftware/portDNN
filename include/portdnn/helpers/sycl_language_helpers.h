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

#ifndef INCLUDE_PORTDNN_HELPERS_SYCL_LANGUAGE_HELPERS_H_
#define INCLUDE_PORTDNN_HELPERS_SYCL_LANGUAGE_HELPERS_H_

#include <CL/sycl.hpp>

namespace sycldnn {
namespace helpers {

// Vary aliasing to handle template differences between SYCL-2020 and SYCL-1.2.1
#ifdef SYCL_IMPLEMENTATION_ONEAPI
template <typename T>
using buffer_allocator = cl::sycl::buffer_allocator<T>;
#else
// Define template for SYCL-1.2.1 but ignore it
template <typename T>
using buffer_allocator = cl::sycl::buffer_allocator;
#endif

}  // namespace helpers
}  // namespace sycldnn

#endif  // INCLUDE_PORTDNN_HELPERS_SYCL_LANGUAGE_HELPERS_H_
