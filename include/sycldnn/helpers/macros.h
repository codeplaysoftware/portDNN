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

/**
 * \file
 * Provides various helper macros.
 */

/** Standard stringification macros */
#define SNN_MAKE_STRING(x) SNN_MAKE_STRING_IMPL(x)

/**
 * \def SNN_MAKE_STRING_IMPL
 * Implementation detail. Required for correct stringification.
 */
#define SNN_MAKE_STRING_IMPL(x) #x

/**
 * Silences unused parameter warnings.
 */
#define SNN_UNUSED_VAR(x) (void)x;

/**
 * Provide a dummy attribute check for compilers which don't provide one. Always
 * returns false for compilers without __has_attribute() support.
 */
#ifdef __has_attribute
#define SNN_HAS_ATTRIBUTE(x) __has_attribute(x)
#else
#define SNN_HAS_ATTRIBUTE(x) 0
#endif

/**
 * Provides an always inline attribute to use for device code. This ensures that
 * all functions in the kernel are inlined, so the optimiser can better
 * understand the whole kernel.
 */
#if defined(__SYCL_DEVICE_ONLY__) && SNN_HAS_ATTRIBUTE(always_inline)
#define SNN_ALWAYS_INLINE __attribute__((always_inline))
#else
#define SNN_ALWAYS_INLINE
#endif

/**
 * Suggests to the compiler to unroll loops, typically on the device this leads
 * to performance benefits, but make sure that this is benchmarked.
 */
#if defined(__SYCL_DEVICE_ONLY__) && defined(__clang__)
#define SNN_PRAGMA_UNROLL \
  _Pragma("clang loop unroll(enable) interleave(enable)")
#else
#define SNN_PRAGMA_UNROLL
#endif

/**
 * \def SNN_ASSERT
 * Optional assert macro. Can be redefined by the user if required.
 */
#ifndef SNN_ASSERT
// By default don't assert on the device and use standard assert on the host.
#ifdef __SYCL_DEVICE_ONLY__
#define SNN_ASSERT(condition, message)
#else
#include <cassert>
#define SNN_ASSERT(condition, message) assert((condition) && (message))
#endif  // __SYCL_DEVICE_ONLY__
#endif  // SNN_ASSERT

/**
 * Validate that a condition is always true. Throw an assert or return
 * InvalidParameter if not.
 * \param condition Condition to check.
 * \param message   Error message to give if condition not met.
 */
#define SNN_VALIDATE_PARAM(condition, message)            \
  do {                                                    \
    SNN_ASSERT(condition, message);                       \
    if (!(condition)) {                                   \
      return SNNStatus{{}, StatusCode::InvalidParameter}; \
    }                                                     \
  } while (0)

#endif  // SYCLDNN_INCLUDE_HELPER_MACROS_H_
