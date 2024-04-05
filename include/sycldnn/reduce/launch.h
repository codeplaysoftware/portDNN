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
#ifndef SYCLDNN_INCLUDE_REDUCE_LAUNCH_H_
#define SYCLDNN_INCLUDE_REDUCE_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::reduce::launch() function, which asynchronously
 * dispatches the SYCL kernels required to perform reductions.
 */
#include <type_traits>

#include "sycldnn/backend/backend_helpers.h"
#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/internal/reduce/launch.h"
#include "sycldnn/reduce/operators.h"
#include "sycldnn/reduce/params.h"

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
 * Launch a reduction of dims[n] applying Op on the axes[m] dimensions. The
 * output shape is dims[n] with dims[axes[m]] = 1.
 *
 * \tparam Op Operation to apply on the reduced dimension
 * \param input A pointer to the memory representing the input tensor.
 * \param output A pointer to the memory representing the output tensor.
 * \param params NDReduceParams struct containing dimensions of tensor, which
 *           dimensions are to be reduced and is the output tensor is
 *           initialised
 * \param backend The backend implementation, used to map between
 *                pointer representations.
 * \param workspace A pointer workspace memory to be used in cases where there
 *                  are more than 1 non consecutive axes of reduction
 * \param workspace_size Number of elements to of workspace use helper function
 *                       to query
 * \return Returns an SNNStatus containing the SYCL event tied to the
 *         kernel launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, typename Op, typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_buffer_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T> output,
                 NDReduceParams const& params, Backend& backend,
                 typename Backend::template pointer_type<T> workspace,
                 size_t workspace_size) {
  return reduce::internal::sublaunch<T, Op, Backend>(
      input, output, params, backend, workspace, workspace_size, {});
}

#ifdef SNN_ENABLE_USM
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

/**
 * Launch a reduction of dims[n] applying Op on the axes[m] dimensions. The
 * output shape is dims[n] with dims[axes[m]] = 1.
 *
 * \tparam Op Operation to apply on the reduced dimension
 * \param input A pointer to the memory representing the input tensor.
 * \param output A pointer to the memory representing the output tensor.
 * \param rp NDReduceParams struct containing dimensions of tensor, which
 *           dimensions are to be reduced and is the output tensor is
 *           initialised
 * \param backend The backend implementation, used to map between
 *                pointer representations.
 * \param workspace A pointer workspace memory to be used in cases where there
 *                  are more than 1 non consecutive axes of reduction
 * \param workspace_size Number of elements to of workspace use helper function
 *                       to query
 * \param events     Events which should be completed before the operation
 * \return Returns an SNNStatus containing the SYCL event tied to the
 *         kernel launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, typename Op, typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_usm_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T> output,
                 NDReduceParams const& params, Backend& backend,
                 typename Backend::template pointer_type<T> workspace,
                 size_t workspace_size,
                 const std::vector<cl::sycl::event>& events = {}) {
  return reduce::internal::sublaunch<T, Op, Backend>(
      input, output, params, backend, workspace, workspace_size, events);
}
#endif
}  // namespace reduce
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_REDUCE_LAUNCH_H_
