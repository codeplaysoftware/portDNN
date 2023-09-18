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
#ifndef PORTDNN_INCLUDE_INTERNAL_MATMUL_LAUNCH_H_
#define PORTDNN_INCLUDE_INTERNAL_MATMUL_LAUNCH_H_

#include <CL/sycl.hpp>

#include "portdnn/mem_object.h"
#include "portdnn/status.h"

#include "portdnn/matmul/params.h"

#include "portdnn/export.h"

namespace sycldnn {
namespace matmul {
namespace internal {

/**
 * The internal matrix multiply launcher.
 *
 * Implemented in the compiled SYCL DNN library.
 */
template <typename T, bool TransposeLHS, bool TransposeRHS,
          template <typename> class MemObj>
SNN_EXPORT SNNStatus launch(MemObj<T const>& lhs, MemObj<T const>& rhs,
                            MemObj<T>& output, MatmulParams const& params,
                            cl::sycl::queue& queue,
                            const std::vector<cl::sycl::event>& events);

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
template <typename T, bool TransposeLHS, bool TransposeRHS, typename Backend>
SNNStatus sublaunch(typename Backend::template pointer_type<T const> lhs,
                    typename Backend::template pointer_type<T const> rhs,
                    typename Backend::template pointer_type<T> output,
                    MatmulParams const& params, Backend& backend,
                    const std::vector<cl::sycl::event>& events = {}) {
  SNN_VALIDATE_PARAM(params.batches > 0,
                     "The number of batches must be positive.");
  SNN_VALIDATE_PARAM(params.m > 0, "The value of m must be positive.");
  SNN_VALIDATE_PARAM(params.k > 0, "The value of k must  be positive.");
  SNN_VALIDATE_PARAM(params.n > 0, "The value of n must be positive.");

  size_t lhs_size = params.batches * params.m * params.k;
  size_t rhs_size = params.batches * params.k * params.n;
  size_t out_size = params.batches * params.m * params.n;

  auto lhs_acc = backend.get_mem_object(lhs, lhs_size);
  auto rhs_acc = backend.get_mem_object(rhs, rhs_size);
  auto out_acc = backend.get_mem_object(output, out_size);

  auto sycl_queue = backend.get_queue();

  return internal::launch<T, TransposeLHS, TransposeRHS>(
      lhs_acc, rhs_acc, out_acc, params, sycl_queue, events);
}

}  // namespace internal
}  // namespace matmul
}  // namespace sycldnn

#endif  // PORTDNN_INCLUDE_INTERNAL_MATMUL_LAUNCH_H_
