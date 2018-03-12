/*
 * Copyright 2018 Codeplay Software Ltd
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
#ifndef SYCLDNN_INCLUDE_HELPER_MACROS_H_
#define SYCLDNN_INCLUDE_HELPER_MACROS_H_
// Standard stringification macros
#define SNN_MAKE_STRING(x) SNN_MAKE_STRING_IMPL(x)
#define SNN_MAKE_STRING_IMPL(x) #x
// Provide a dummy attribute check for compilers which don't provide one.
#ifdef __has_attribute
#define SNN_HAS_ATTRIBUTE(x) __has_attribute(x)
#else
#define SNN_HAS_ATTRIBUTE(x) 0
#endif
// Provide an always inline attribute to use for device code. This ensures that
// all functions in the kernel are inlined, so the optimiser can better
// understand the whole kernel.
#if defined(__SYCL_DEVICE_ONLY__) && SNN_HAS_ATTRIBUTE(always_inline)
#define SNN_ALWAYS_INLINE __attribute__((always_inline))
#else
#define SNN_ALWAYS_INLINE
#endif
// Suggest to the compiler to unroll loops, typically on the device this leads
// to performance benefits, but make sure that this is benchmarked.
#if defined(__SYCL_DEVICE_ONLY__) && defined(__CLANG__)
#define SNN_PRAGMA_UNROLL \
  _Pragma("clang loop unroll(enable) interleave(enable)")
#else
#define SNN_PRAGMA_UNROLL
#endif
#endif  // SYCLDNN_INCLUDE_HELPER_MACROS_H_
