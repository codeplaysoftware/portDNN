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
#ifndef PORTDNN_INCLUDE_HELPER_CL_MACROS_H_
#define PORTDNN_INCLUDE_HELPER_CL_MACROS_H_

#include "portdnn/helpers/macros.h"

#include <CL/cl.h>

/**
 * \file
 * Provides helper macros for calling CL functions.
 */

namespace sycldnn {
namespace helpers {

/**
 * Convert an OpenCL error to a string describing the error, as specified in the
 * OpenCL spec.
 *
 * \param error The OpenCL error to convert.
 * \return A string describing the given error.
 */
inline char const* ocl_error_message(cl_int error) {
/** Internal macro to handle OpenCL error case.  */
#define SNN_INTERNAL_ERR_CASE(CL_ERROR) \
  case CL_ERROR:                        \
    return #CL_ERROR

  switch (error) {
    SNN_INTERNAL_ERR_CASE(CL_DEVICE_NOT_FOUND);
    SNN_INTERNAL_ERR_CASE(CL_DEVICE_NOT_AVAILABLE);
    SNN_INTERNAL_ERR_CASE(CL_COMPILER_NOT_AVAILABLE);
    SNN_INTERNAL_ERR_CASE(CL_MEM_OBJECT_ALLOCATION_FAILURE);
    SNN_INTERNAL_ERR_CASE(CL_OUT_OF_RESOURCES);
    SNN_INTERNAL_ERR_CASE(CL_OUT_OF_HOST_MEMORY);
    SNN_INTERNAL_ERR_CASE(CL_PROFILING_INFO_NOT_AVAILABLE);
    SNN_INTERNAL_ERR_CASE(CL_MEM_COPY_OVERLAP);
    SNN_INTERNAL_ERR_CASE(CL_IMAGE_FORMAT_MISMATCH);
    SNN_INTERNAL_ERR_CASE(CL_IMAGE_FORMAT_NOT_SUPPORTED);
    SNN_INTERNAL_ERR_CASE(CL_BUILD_PROGRAM_FAILURE);
    SNN_INTERNAL_ERR_CASE(CL_MAP_FAILURE);
#ifdef CL_VERSION_1_1
    SNN_INTERNAL_ERR_CASE(CL_MISALIGNED_SUB_BUFFER_OFFSET);
    SNN_INTERNAL_ERR_CASE(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
#endif  // CL_VERSION_1_1
#ifdef CL_VERSION_1_2
    SNN_INTERNAL_ERR_CASE(CL_COMPILE_PROGRAM_FAILURE);
    SNN_INTERNAL_ERR_CASE(CL_LINKER_NOT_AVAILABLE);
    SNN_INTERNAL_ERR_CASE(CL_LINK_PROGRAM_FAILURE);
    SNN_INTERNAL_ERR_CASE(CL_DEVICE_PARTITION_FAILED);
    SNN_INTERNAL_ERR_CASE(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
#endif  // CL_VERSION_1_2
    SNN_INTERNAL_ERR_CASE(CL_INVALID_VALUE);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_DEVICE_TYPE);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_PLATFORM);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_DEVICE);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_CONTEXT);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_QUEUE_PROPERTIES);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_COMMAND_QUEUE);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_HOST_PTR);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_MEM_OBJECT);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_IMAGE_SIZE);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_SAMPLER);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_BINARY);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_BUILD_OPTIONS);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_PROGRAM);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_PROGRAM_EXECUTABLE);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_KERNEL_NAME);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_KERNEL_DEFINITION);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_KERNEL);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_ARG_INDEX);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_ARG_VALUE);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_ARG_SIZE);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_KERNEL_ARGS);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_WORK_DIMENSION);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_WORK_GROUP_SIZE);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_WORK_ITEM_SIZE);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_GLOBAL_OFFSET);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_EVENT_WAIT_LIST);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_EVENT);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_OPERATION);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_GL_OBJECT);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_BUFFER_SIZE);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_MIP_LEVEL);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_GLOBAL_WORK_SIZE);
#ifdef CL_VERSION_1_1
    SNN_INTERNAL_ERR_CASE(CL_INVALID_PROPERTY);
#endif  // CL_VERSION_1_1
#ifdef CL_VERSION_1_2
    SNN_INTERNAL_ERR_CASE(CL_INVALID_IMAGE_DESCRIPTOR);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_COMPILER_OPTIONS);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_LINKER_OPTIONS);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_DEVICE_PARTITION_COUNT);
#endif  // CL_VERSION_1_2
#ifdef CL_VERSION_2_0
    SNN_INTERNAL_ERR_CASE(CL_INVALID_PIPE_SIZE);
    SNN_INTERNAL_ERR_CASE(CL_INVALID_DEVICE_QUEUE);
#endif  // CL_VERSION_2_0
#ifdef CL_VERSION_2_2
    SNN_INTERNAL_ERR_CASE(CL_INVALID_SPEC_ID);
    SNN_INTERNAL_ERR_CASE(CL_MAX_SIZE_RESTRICTION_EXCEEDED);
#endif  // CL_VERSION_2_2
    default:
      return "Unknown OpenCL error";
#undef SNN_INTERNAL_ERR_CASE
  }
}

}  // namespace helpers
}  // namespace sycldnn

/**
 * Macro that will check a value against a known-good value, to check
 * whether any errors have happened. Using this macro requires the headers
 * stdexcept and string.
 *
 * \param err The variable to check against CL_SUCCESS
 */
#define SNN_CL_CHECK_ERR(err)                                           \
  if (CL_SUCCESS != err) {                                              \
    throw std::runtime_error(                                           \
        SNN_ERROR_MSG(OpenCL, sycldnn::helpers::ocl_error_message(err), \
                      __FILE__, __LINE__));                             \
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

#endif  // PORTDNN_INCLUDE_HELPER_CL_MACROS_H_
