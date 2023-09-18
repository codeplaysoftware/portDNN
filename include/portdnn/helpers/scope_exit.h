/*
 * Copyright Codeplay Software Ltd.
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
#ifndef PORTDNN_INCLUDE_HELPERS_SCOPE_EXIT_H_
#define PORTDNN_INCLUDE_HELPERS_SCOPE_EXIT_H_

#include <utility>

namespace sycldnn {
namespace helpers {
/** Wrapper around a task which will execute on destruction. */
template <typename Function>
struct scope_exit_task {
  /** Function object to call on destruction. */
  Function f;
  /** Destructor which executes the task. */
  ~scope_exit_task() { f(); }
};
/** Factory to produce `scope_exit_task`s from a lambda. */
struct scope_exit_task_creator {
  /**
   * Operator to convert a lambda into a scope_exit_task.
   * \param f Function object to use to create a scope_exit_task.
   * \return A scope_exit_task which calls f on destruction.
   */
  template <typename Function>
  scope_exit_task<Function> operator+(Function&& f) {
    return scope_exit_task<Function>{std::move(f)};
  }
};
}  // namespace helpers
}  // namespace sycldnn

/**
 * \def SNN_ON_SCOPE_EXIT
 * Macro to create tasks to be executed when leaving the current scope:
 * \code
 *    SNN_ON_SCOPE_EXIT {
 *      // Anything here will only be executed at scope exit
 *    };
 * \endcode
 */
#define SNN_ON_SCOPE_EXIT_IMPL1(TAG) \
  auto scope_exit_task_##TAG =       \
      ::sycldnn::helpers::scope_exit_task_creator{} + [&]()
#define SNN_ON_SCOPE_EXIT_IMPL(TAG) SNN_ON_SCOPE_EXIT_IMPL1(TAG)
#define SNN_ON_SCOPE_EXIT SNN_ON_SCOPE_EXIT_IMPL(__LINE__)

#endif  // PORTDNN_INCLUDE_HELPERS_SCOPE_EXIT_H_
