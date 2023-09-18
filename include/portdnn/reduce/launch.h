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
#ifndef PORTDNN_INCLUDE_REDUCE_LAUNCH_H_
#define PORTDNN_INCLUDE_REDUCE_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::reduce::launch() function, which asynchronously
 * dispatches the SYCL kernels required to perform reductions.
 */
#include <type_traits>

#include "portdnn/backend/backend_helpers.h"
#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/internal/helpers/types.h"
#include "portdnn/internal/reduce/launch.h"
#include "portdnn/reduce/operators.h"

namespace sycldnn {
namespace reduce {
/**
 * Launch a reduction of [batch, outer, inner] applying Op on the outer
 * dimension. The output shape is [batch, inner].
 *
 * \tparam Op Operation to apply on the reduced dimension
 * \param input A pointer to the memory representing the input tensor.
 * \param output A pointer to the memory representing the output tensor.
 * \param batches The number of batches. Must be a positive value.
 * \param outer Outer size. This is the dimension that is always reduced. Must
 * be a positive value.
 * \param inner Inner size. Must be a positive value.
 * \param backend The backend implementation, used to map between pointer
 *                representations.
 * \return Returns an SNNStatus containing the SYCL event tied to the kernel
 *         launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, typename Op, typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_buffer_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T> output, int batches,
                 int outer, int inner, Backend& backend) {
  return internal::sublaunch<T, Op, Backend>(input, output, batches, outer,
                                             inner, backend, {});
}

/**
 * Launch a reduction of [batch, outer, inner] applying Op on the outer
 * dimension. The output shape is [batch, inner].
 *
 * \tparam Op Operation to apply on the reduced dimension
 * \param input A pointer to the memory representing the input tensor.
 * \param output A pointer to the memory representing the output tensor.
 * \param batches The number of batches. Must be a positive value.
 * \param outer Outer size. This is the dimension that is always reduced. Must
 * be a positive value.
 * \param inner Inner size. Must be a positive value.
 * \param backend The backend implementation, used to map between pointer
 *                representations.
 * \param events     Events which should be completed before the operation
 * \return Returns an SNNStatus containing the SYCL event tied to the kernel
 *         launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, typename Op, typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_usm_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T> output, int batches,
                 int outer, int inner, Backend& backend,
                 const std::vector<cl::sycl::event>& events = {}) {
  return internal::sublaunch<T, Op, Backend>(input, output, batches, outer,
                                             inner, backend, events);
}
}  // namespace reduce
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_REDUCE_LAUNCH_H_
