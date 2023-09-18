
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
#ifndef PORTDNN_INCLUDE_BACKEND_HELPERS_H_
#define PORTDNN_INCLUDE_BACKEND_HELPERS_H_

#include <type_traits>

namespace sycldnn {
namespace backend {
namespace internal {
template <typename Backend>
struct InternalBackend;
}
// Forward declerations
struct SNNBackend;
struct SNNUSMBackend;
struct SyclBLASBackend;
struct CLBlasBackend;
struct EigenBackend;

// Helpers to check if backend uses USM or buffers
template <typename Backend>
struct is_usm_backend
    : std::integral_constant<
          bool,
          std::is_same<Backend, SNNUSMBackend>::value ||
              std::is_same<Backend,
                           internal::InternalBackend<SNNUSMBackend>>::value> {};

template <typename Backend>
inline constexpr bool is_usm_backend_v = is_usm_backend<Backend>::value;

template <typename Backend>
struct is_buffer_backend
    : std::integral_constant<
          bool,
          std::is_same<Backend, SNNBackend>::value ||
              std::is_same<Backend, SyclBLASBackend>::value ||
              std::is_same<Backend, CLBlasBackend>::value ||
              std::is_same<Backend, EigenBackend>::value ||
              std::is_same<Backend,
                           internal::InternalBackend<SNNBackend>>::value ||
              std::is_same<Backend,
                           internal::InternalBackend<SyclBLASBackend>>::value ||
              std::is_same<Backend,
                           internal::InternalBackend<CLBlasBackend>>::value ||
              std::is_same<Backend,
                           internal::InternalBackend<EigenBackend>>::value> {};

template <typename Backend>
inline constexpr bool is_buffer_backend_v = is_buffer_backend<Backend>::value;

template <typename Backend>
struct supports_interleaved_matmul
    : std::integral_constant<bool,
                             std::is_same<Backend, SyclBLASBackend>::value> {};

}  // namespace backend
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_BACKEND_HELPERS_H_
