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

#ifndef SYCLDNN_INCLUDE_POOLING_LAUNCH_H_
#define SYCLDNN_INCLUDE_POOLING_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::pooling::launch() function, which asynchronously
 * dispatches the SYCL kernels to compute a 2D pooling operation.
 */

#include "sycldnn/accessor_types.h"
#include "sycldnn/internal/pooling/launch_internal.h"
#include "sycldnn/pooling/params.h"
#include "sycldnn/status.h"

namespace sycldnn {
/** Namespace containing all pooling operations. */
namespace pooling {

/**
 * Launch the pooling operation kernel.
 *
 * \tparam T         The data type of the input tensor.
 * \tparam PoolType  The type of pooling used depends on the PoolType template
 *                   parameter, which can be used to specify either Max or
 *                   Average pooling.
 * \tparam Direction Whether the pooling operation computed should be the
 *                   Forward or Backpropagate pass.
 * \tparam Backend   The type of the Backend.
 *
 * \param [in]  input    A pointer to the input tensor.
 * \param [out] output   A pointer to the output tensor.
 * \param [in]  pp       The parameters of the pooling operation.
 * \param [in]  backend  The backend that provides access to the SYCL buffers
 *                       corresponding to the input and output pointers.
 * \return An SNNStatus containing the SYCL event tied to the kernel launches
 *         and a StatusCode enum showing if the launch was OK or whether it
 *         encountered some problem.
 */
template <typename T, template <typename U> class PoolType, typename Direction,
          typename Backend>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T> output,
                 const PoolingParams& pp, Backend& backend) {
  SNN_VALIDATE_PARAM(pp.batch > 0, "The batch size must be positive.");
  SNN_VALIDATE_PARAM(pp.channels > 0,
                     "The number of channels must be positive.");
  SNN_VALIDATE_PARAM(pp.in_rows > 0,
                     "The number of input rows must be positive.");
  SNN_VALIDATE_PARAM(pp.in_cols > 0,
                     "The number of input columns must be positive.");
  SNN_VALIDATE_PARAM(pp.out_rows > 0,
                     "The number of output rows must be positive.");
  SNN_VALIDATE_PARAM(pp.out_cols > 0,
                     "The number of output columns must be positive.");
  SNN_VALIDATE_PARAM(pp.window_rows > 0,
                     "The number of window rows must be positive.");
  SNN_VALIDATE_PARAM(pp.window_cols > 0,
                     "The number of window columns must be positive.");
  SNN_VALIDATE_PARAM(pp.stride_rows > 0,
                     "The stride in the row direction must be positive.");
  SNN_VALIDATE_PARAM(pp.stride_cols > 0,
                     "The stride in the column direction must be positive.");
  SNN_VALIDATE_PARAM(pp.pad_rows >= 0,
                     "The padding in the row direction must be non-negative.");
  SNN_VALIDATE_PARAM(
      pp.pad_cols >= 0,
      "The padding in the column direction must be non-negative.");

  size_t const input_size = pp.batch * pp.in_rows * pp.in_cols * pp.channels;
  size_t const output_size = pp.batch * pp.out_rows * pp.out_cols * pp.channels;

  auto inp_buf = backend.get_buffer(input, input_size);
  auto outp_buf = backend.get_buffer(output, output_size);

  auto const inp_offset = backend.get_offset(input);
  auto const outp_offset = backend.get_offset(output);

  ReadAccessor<T const> inp_access{inp_buf, cl::sycl::range<1>{input_size},
                                   cl::sycl::id<1>{inp_offset}};
  WriteAccessor<T> outp_access{outp_buf, cl::sycl::range<1>{output_size},
                               cl::sycl::id<1>{outp_offset}};

  auto queue = backend.get_queue();
  return internal::launch_pooling<T, PoolType, Direction>(
      inp_access, outp_access, pp, queue);
}

}  // namespace pooling
}  // namespace sycldnn
#endif  // SYCLDNN_INCLUDE_POOLING_LAUNCH_H_
