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
#ifndef PORTDNN_INCLUDE_BACKEND_BACKEND_TRAITS_H_
#define PORTDNN_INCLUDE_BACKEND_BACKEND_TRAITS_H_

namespace sycldnn {
namespace backend {

/**
 * Traits class which contains the pointer aliases for the Backend.
 *
 * This should be specialised for every Backend, and each specialisation should
 * provide:
 *
 *  - an alias for pointer_type<T> which gives the pointer type for a data type
 * T used in the external interface of portDNN.
 *  - an alias for internal_pointer_type<T> which gives the pointer type for a
 * data type T used internally and used in the matmul interface of the backend.
 */
template <typename Backend>
struct BackendTraits;

}  // namespace backend
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_BACKEND_BACKEND_TRAITS_H_
