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
#include "sycldnn/status.h"

#include "sycldnn/helpers/macros.h"

#include "sycldnn/pooling/params.h"
#include "sycldnn/pooling/sizes.h"

#include "sycldnn/internal/pooling/launch_internal.h"

namespace sycldnn {
/** Namespace containing all pooling operations. */
namespace pooling {
/** Namespace containing internal implementation details for pooling. */
namespace internal {

/**
 * Validate that the user provided pooling parameters are consistent with what
 * is expected by SYCL-DNN.
 *
 * If compiled with asserts, any invalid parameter will fail an assert.
 * Otherwise a status code \ref StatusCode::InvalidParameter will be returned.
 *
 * \param [in] params User provided parameters to validate
 * \return An SNNStatus obejct containing either \ref StatusCode::OK if all
 *         parameters are valid, or \ref StatusCode::InvalidParameter otherwise.
 */
SNNStatus inline validate_params(PoolingParams const& params) {
  SNN_VALIDATE_PARAM(params.batch > 0, "The batch size must be positive.");
  SNN_VALIDATE_PARAM(params.channels > 0,
                     "The number of channels must be positive.");
  SNN_VALIDATE_PARAM(params.in_rows > 0,
                     "The number of input rows must be positive.");
  SNN_VALIDATE_PARAM(params.in_cols > 0,
                     "The number of input columns must be positive.");
  SNN_VALIDATE_PARAM(params.out_rows > 0,
                     "The number of output rows must be positive.");
  SNN_VALIDATE_PARAM(params.out_cols > 0,
                     "The number of output columns must be positive.");
  SNN_VALIDATE_PARAM(params.window_rows > 0,
                     "The number of window rows must be positive.");
  SNN_VALIDATE_PARAM(params.window_cols > 0,
                     "The number of window columns must be positive.");
  SNN_VALIDATE_PARAM(params.stride_rows > 0,
                     "The stride in the row direction must be positive.");
  SNN_VALIDATE_PARAM(params.stride_cols > 0,
                     "The stride in the column direction must be positive.");
  SNN_VALIDATE_PARAM(params.pad_rows >= 0,
                     "The padding in the row direction must be non-negative.");
  SNN_VALIDATE_PARAM(
      params.pad_cols >= 0,
      "The padding in the column direction must be non-negative.");
  return SNNStatus{{}, StatusCode::OK};
}

}  // namespace internal

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
 * \return An \ref SNNStatus containing the SYCL event tied to the kernel
 *         launches and a \ref StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, template <typename> class PoolType, typename Direction,
          typename Backend,
          typename = internal::DisableIfMaxGradient<T, PoolType, Direction>>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T> output,
                 const PoolingParams& pp, Backend& backend) {
  auto validation_status = internal::validate_params(pp);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }
  auto sizes = get_sizes<Direction>(pp);

  auto inp_buf = backend.get_buffer(input, sizes.input_size);
  auto outp_buf = backend.get_buffer(output, sizes.output_size);

  auto const inp_offset = backend.get_offset(input);
  auto const outp_offset = backend.get_offset(output);

  ReadAccessor<T const> inp_access{inp_buf,
                                   cl::sycl::range<1>{sizes.input_size},
                                   cl::sycl::id<1>{inp_offset}};
  WriteAccessor<T> outp_access{outp_buf, cl::sycl::range<1>{sizes.output_size},
                               cl::sycl::id<1>{outp_offset}};

  auto queue = backend.get_queue();
  return internal::launch_pooling<T, PoolType, Direction>(
      inp_access, outp_access, pp, queue);
}

/**
 * Launch the max pooling gradient kernel.
 *
 * \tparam T         The data type of the input tensor.
 * \tparam PoolType  The type of pooling used depends on the PoolType template
 *                   parameter, which can be used to specify either Max or
 *                   Average pooling.
 * \tparam Direction Whether the pooling operation computed should be the
 *                   Forward or Backpropagate pass.
 * \tparam Backend   The type of the Backend.
 *
 * \param [in]  input_data     A pointer to the original input tensor.
 * \param [in]  output_data    A pointer to the original output tensor.
 * \param [in]  input_backprop A pointer to the backprop error tensor.
 * \param [out] output         A pointer to the output tensor.
 * \param [in]  pp             The parameters of the pooling operation.
 * \param [in]  backend        The backend that provides access to the SYCL
 *                             buffers corresponding to the input and output
 *                             pointers.
 * \return An \ref SNNStatus containing the SYCL event tied to the kernel
 *         launches and a \ref StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, template <typename> class PoolType, typename Direction,
          typename Backend,
          typename = internal::EnableIfMaxGradient<T, PoolType, Direction>>
SNNStatus launch(
    typename Backend::template pointer_type<T const> input_data,
    typename Backend::template pointer_type<T const> output_data,
    typename Backend::template pointer_type<T const> input_backprop,
    typename Backend::template pointer_type<T> output, const PoolingParams& pp,
    Backend& backend) {
  auto validation_status = internal::validate_params(pp);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }
  auto fwd_sizes = get_sizes<Forward>(pp);
  auto back_sizes = get_sizes<Backpropagate>(pp);

  auto inp_data_buf = backend.get_buffer(input_data, fwd_sizes.input_size);
  auto outp_data_buf = backend.get_buffer(output_data, fwd_sizes.output_size);
  auto inp_backprop_buf =
      backend.get_buffer(input_backprop, back_sizes.input_size);
  auto outp_backprop_buf = backend.get_buffer(output, back_sizes.output_size);

  auto const inp_data_offset = backend.get_offset(input_data);
  auto const outp_data_offset = backend.get_offset(output_data);
  auto const inp_backprop_offset = backend.get_offset(input_backprop);
  auto const outp_backprop_offset = backend.get_offset(output);

  ReadAccessor<T const> inp_data_access{
      inp_data_buf, cl::sycl::range<1>{fwd_sizes.input_size},
      cl::sycl::id<1>{inp_data_offset}};
  ReadAccessor<T const> outp_data_access{
      outp_data_buf, cl::sycl::range<1>{fwd_sizes.output_size},
      cl::sycl::id<1>{outp_data_offset}};
  ReadAccessor<T const> inp_backprop_access{
      inp_backprop_buf, cl::sycl::range<1>{back_sizes.input_size},
      cl::sycl::id<1>{inp_backprop_offset}};
  WriteAccessor<T> outp_backprop_access{
      outp_backprop_buf, cl::sycl::range<1>{back_sizes.output_size},
      cl::sycl::id<1>{outp_backprop_offset}};

  auto queue = backend.get_queue();
  return internal::launch_pooling<T, PoolType, Direction>(
      inp_data_access, outp_data_access, inp_backprop_access,
      outp_backprop_access, pp, queue);
}

}  // namespace pooling
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_POOLING_LAUNCH_H_
