/*
 * Copyright 2019 Codeplay Software Ltd
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
#ifndef SYCLDNN_INCLUDE_HELPER_CL_MACROS_H_
#define SYCLDNN_INCLUDE_HELPER_CL_MACROS_H_

#include "sycldnn/helpers/macros.h"

/**
 * \file
 * Provides helper macros for calling CL functions.
 */

/**
 * Macro that will check a value against a known-good value, to check
 * whether any errors have happened. Using this macro requires the headers
 * stdexcept and string.
 *
 * \param err The variable to check against CL_SUCCESS
 */
#define SNN_CL_CHECK_ERR(err)                                                 \
  if (CL_SUCCESS != err) {                                                    \
    throw std::runtime_error(SNN_ERROR_MSG(OpenCL, err, __FILE__, __LINE__)); \
  }

/**
 * Helper for calling OpenCL functions with an error variable as a
 * parameter. Use as `auto result = SNN_CL_CALL_WITH_ERR();`.
 * The args are passed to the function by __VA_ARGS__.
 *
 * \param fun The OpenCL function to call
 * \param err The error variable to be passed by pointer to fun
 */
#define SNN_CL_CALL_WITH_ERR(fun, err, ...) \
  (fun)(__VA_ARGS__, &err);                 \
  SNN_CL_CHECK_ERR(err);

/**
 * Helper for calling OpenCL functions which return errors.
 * Use as `SNN_CL_CALL_RETURN_ERR(clFunction, err, ...);`.
 * The args are passed to the function by __VA_ARGS__.
 *
 * \param fun The OpenCL function to call
 * \param err The error variable to be assigned to by fun
 */
#define SNN_CL_CALL_RETURN_ERR(fun, err, ...) \
  err = (fun)(__VA_ARGS__);                   \
  SNN_CL_CHECK_ERR(err);

#endif  // SYCLDNN_INCLUDE_HELPER_CL_MACROS_H_
