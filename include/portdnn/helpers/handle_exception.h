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

#ifndef INCLUDE_PORTDNN_HELPERS_HANDLE_EXCEPTION_H_
#define INCLUDE_PORTDNN_HELPERS_HANDLE_EXCEPTION_H_

#include <CL/sycl.hpp>

#include <exception>
#include <string>
#include <type_traits>

/**
 * \file
 * Defines helper function for handling exceptions in benchmarks and tests.
 */

namespace sycldnn {
namespace helpers {

/**
 * Struct that chooses correct string name for a given exception type.
 * \tparam Exception The exception type whose name is required
 */
template <typename Exception>
struct ExceptionName {
 private:
  struct not_implemented;
  static_assert(
      std::is_same<not_implemented, Exception>::value,
      "ExceptionName can only be instantiated with std or SYCL exceptions");
};

/**
 * Specialisation of ExceptionName for std::exception.
 */
template <>
struct ExceptionName<std::exception> {
  /** Member variable string representation of template parameter name. */
  static constexpr auto value = "std::exception";
};

/**
 * Specialisation of ExceptionName for cl::sycl::exception.
 */
template <>
struct ExceptionName<cl::sycl::exception> {
  /** Member variable string representation of template parameter name. */
  static constexpr auto value = "cl::sycl::exception";
};

/**
 * Helper function that formats a string to describe what has gone wrong,
 * then passes that string to a user action.
 * \tparam Ex the exception type - std:: or cl::sycl::
 * \tparam Func an Invocable type
 * \param e The exception being handled
 * \param f the Invocable to be invoked after string processing
 */
template <typename Ex, typename Func>
inline void handle_exception(Ex const& e, Func&& f) {
  auto message =
      std::string{ExceptionName<Ex>::value} + " caught: " + e.what() + ". ";
  f(message);
}

}  // namespace helpers
}  // namespace sycldnn

#endif  // INCLUDE_PORTDNN_HELPERS_HANDLE_EXCEPTION_H_
