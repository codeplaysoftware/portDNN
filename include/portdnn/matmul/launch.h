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
#ifndef PORTDNN_INCLUDE_MATMUL_LAUNCH_H_
#define PORTDNN_INCLUDE_MATMUL_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::matmul::launch() function, which asynchronously
 * dispatches the SYCL kernels required to perform a matrix multiply.
 */
#include "portdnn/backend/backend_helpers.h"
#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/helpers/macros.h"
#include "portdnn/internal/matmul/launch.h"
#include "portdnn/matmul/params.h"

namespace sycldnn {
namespace matmul {
/**
 * Launch a batched matrix multiplication.
 *
 * Will compute: output[i] = beta * output[i] + op(lhs[i]) * op(rhs[i])
 * where i ranges over the number of batches and op(X) is either X or X^T if
 * TransposeX is true.
 *
 * \param lhs A pointer to the memory representing the left hand matrix.
 * \param rhs A pointer to the memory representing the right hand matrix.
 * \param output A pointer to the memory representing the output tensor.
 * \param params The parameters of the matrix multiplication operation.
 * \param backend The backend implementation, used to map between pointer
 *                representations.
 * \return Returns an SNNStatus containing the SYCL event tied to the kernel
 *         launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, bool TransposeLHS, bool TransposeRHS, typename Backend,
          typename = typename std::enable_if<
              !sycldnn::backend::is_usm_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> lhs,
                 typename Backend::template pointer_type<T const> rhs,
                 typename Backend::template pointer_type<T> output,
                 MatmulParams const& params, Backend& backend) {
  return internal::sublaunch<T, TransposeLHS, TransposeRHS>(lhs, rhs, output,
                                                            params, backend);
}

/**
 * Launch a batched matrix multiplication.
 *
 * Will compute: output[i] = beta * output[i] + op(lhs[i]) * op(rhs[i])
 * where i ranges over the number of batches and op(X) is either X or X^T if
 * TransposeX is true.
 *
 * \param lhs A pointer to the memory representing the left hand matrix.
 * \param rhs A pointer to the memory representing the right hand matrix.
 * \param output A pointer to the memory representing the output tensor.
 * \param params The parameters of the matrix multiplication operation.
 * \param backend The backend implementation, used to map between pointer
 *                representations.
 * \param events Events which should be completed before the operation
 * \return Returns an SNNStatus containing the SYCL event tied to the kernel
 *         launches and a StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, bool TransposeLHS, bool TransposeRHS, typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_usm_backend_v<Backend>>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> lhs,
                 typename Backend::template pointer_type<T const> rhs,
                 typename Backend::template pointer_type<T> output,
                 MatmulParams const& params, Backend& backend,
                 const std::vector<cl::sycl::event>& events = {}) {
  return internal::sublaunch<T, TransposeLHS, TransposeRHS>(
      lhs, rhs, output, params, backend, events);
}

}  // namespace matmul
}  // namespace sycldnn
#endif  // PORTDNN_INCLUDE_MATMUL_LAUNCH_H_
