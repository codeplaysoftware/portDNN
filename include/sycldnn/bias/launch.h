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

#ifndef SYCLDNN_INCLUDE_BIAS_LAUNCH_H_
#define SYCLDNN_INCLUDE_BIAS_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::bias::launch() function, which asynchronously
 * dispatches the SYCL kernels to compute a bias-add operation.
 */

#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/helpers/macros.h"

#include "sycldnn/bias/params.h"
#include "sycldnn/bias/sizes.h"

#include "sycldnn/internal/bias/launch.h"

namespace sycldnn {
/** Namespace containing all bias-add operations. */
namespace bias {
/** Namespace containing internal implementation details for bias-add. */
namespace internal {

/**
 * Validate that the user provided bias parameters are consistent with what
 * is expected by SYCL-DNN.
 *
 * If compiled with asserts, any invalid parameter will fail an assert.
 * Otherwise a status code \ref StatusCode::InvalidParameter will be returned.
 *
 * \param [in] params User provided parameters to validate
 * \return An SNNStatus obejct containing either \ref StatusCode::OK if all
 *         parameters are valid, or \ref StatusCode::InvalidParameter otherwise.
 */
SNNStatus inline validate_params(BiasParams const& params) {
  SNN_VALIDATE_PARAM(params.batch > 0, "The batch size must be positive.");
  SNN_VALIDATE_PARAM(params.channels > 0,
                     "The number of channels must be positive.");
  SNN_VALIDATE_PARAM(params.bias > 0,
                     "The number of bias values must be positive.");
  SNN_VALIDATE_PARAM(params.in_rows > 0,
                     "The number of input rows must be positive.");
  SNN_VALIDATE_PARAM(params.in_cols > 0,
                     "The number of input columns must be positive.");
  return SNNStatus{{}, StatusCode::OK};
}

}  // namespace internal

/**
 * Launch the bias operation kernel.
 *
 * \tparam T         The data type of the input tensor.
 * \tparam Backend   The type of the Backend.
 *
 * \param [in]  input    A pointer to the input tensor.
 * \param [in]  bias     A pointer to the bias tensor.
 * \param [out] output   A pointer to the output tensor.
 * \param [in]  pp       The parameters of the bias operation.
 * \param [in]  backend  The backend that provides access to the SYCL buffers
 *                       corresponding to the input and output pointers.
 * \return An \ref SNNStatus containing the SYCL event tied to the kernel
 *         launches and a \ref StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, typename Backend>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T const> bias,
                 typename Backend::template pointer_type<T> output,
                 const BiasParams& pp, Backend& backend) {
  auto validation_status = internal::validate_params(pp);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }
  auto sizes = get_sizes(pp);

  auto inp_mem = backend.get_mem_object(input, sizes.input_size);
  auto bias_mem = backend.get_mem_object(bias, sizes.bias_size);
  auto outp_mem = backend.get_mem_object(output, sizes.output_size);
  auto queue = backend.get_queue();
  return internal::launch_bias_add<T>(inp_mem, bias_mem, outp_mem, pp, queue);
}

}  // namespace bias
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_BIAS_LAUNCH_H_
