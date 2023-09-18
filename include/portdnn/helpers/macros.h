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
#ifndef PORTDNN_INCLUDE_HELPER_MACROS_H_
#define PORTDNN_INCLUDE_HELPER_MACROS_H_

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
 * Provide a dummy builtin check for compilers which don't provide one. Always
 * returns false for compilers without __has_builtin() support.
 */
#ifdef __has_builtin
#define SNN_HAS_BUILTIN(x) __has_builtin(x)
#else
#define SNN_HAS_BUILTIN(x) 0
#endif

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
 * Provides an unreachable builtin to tell compiler that any code which reaches
 * the unreachable block is entering undefined behaviour. This allows the
 * compiler to assume that these circumstances never occur and it can optimise
 * around them.
 */
#if SNN_HAS_BUILTIN(__builtin_unreachable)
#define SNN_UNREACHABLE __builtin_unreachable()
#else
#define SNN_UNREACHABLE                                                      \
  SNN_ASSERT(false,                                                          \
             "Code reached an unreachable block, check there are no out of " \
             "bounds accesses.")
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
#define SNN_VALIDATE_PARAM(condition, message) \
  do {                                         \
    SNN_ASSERT(condition, message);            \
    if (!(condition)) {                        \
      return StatusCode::InvalidParameter;     \
    }                                          \
  } while (0)

/**
 * Disable the copy constructor and copy assignment operator for type.
 *
 * This is expected to be used inside the definition of the struct or class of
 * name type.
 * \param type The struct or class name to disable copies for.
 */
#define SNN_DISABLE_COPY(type) \
  type(type const&) = delete;  \
  type& operator=(type const&) = delete

/**
 * Disable the move constructor and move assignment operator for type.
 *
 * This is expected to be used inside the definition of the struct or class of
 * name type.
 * \param type The struct or class name to disable moves for.
 */
#define SNN_DISABLE_MOVE(type) \
  type(type&&) = delete;       \
  type& operator=(type&&) = delete

/**
 * Use the default compiler generated copy constructor and copy assignment
 * operator for type.
 *
 * This is expected to be used inside the definition of the struct or class of
 * name type.
 * \param type The struct or class name to add copy methods for.
 */
#define SNN_DEFAULT_COPY(type) \
  type(type const&) = default; \
  type& operator=(type const&) = default

/**
 * Use the default compiler generated move constructor and move assignment
 * operator for type.
 *
 * This is expected to be used inside the definition of the struct or class of
 * name type.
 * \param type The struct or class name to add move methods for.
 */
#define SNN_DEFAULT_MOVE(type) \
  type(type&&) = default;      \
  type& operator=(type&&) = default

/**
 * Internal macro for formatting error messages used in reporting failures.
 *
 * \param impl Unquoted string indicating where the error came from
 * \param err A string or char* describing the error
 * \param file Should always be __FILE__
 * \param line Should always be __LINE__
 */
#define SNN_ERROR_MSG_IMPL(impl, err, file, line) \
  std::string{#impl " error happened: "} + err + "\nat " #file #line

/**
 * Formats error messages used in reporting failures.
 *
 * \param impl Unquoted string indicating where the error came from
 * \param err A string or char* describing the error
 * \param file Should always be __FILE__
 * \param line Should always be __LINE__
 */
#define SNN_ERROR_MSG(impl, err, file, line) \
  SNN_ERROR_MSG_IMPL(impl, err, file, line)

#ifdef _MSC_VER
/**
 * \def SNN_WINDOWS
 * MSVC platform check macro.
 *
 * `SNN_WINDOWS` will be defined and set when compiling with MSVC, and will not
 * be defined for other compilers.
 */
#define SNN_WINDOWS 1
#endif

#endif  // PORTDNN_INCLUDE_HELPER_MACROS_H_
