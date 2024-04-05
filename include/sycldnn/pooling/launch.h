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

#ifndef SYCLDNN_INCLUDE_POOLING_LAUNCH_H_
#define SYCLDNN_INCLUDE_POOLING_LAUNCH_H_

/**
 * \file
 * Implements the \ref sycldnn::pooling::launch() function, which asynchronously
 * dispatches the SYCL kernels to compute a 2D pooling operation.
 */

#include "sycldnn/backend/backend_helpers.h"
#include "sycldnn/mem_object.h"
#include "sycldnn/status.h"

#include "sycldnn/helpers/macros.h"

#include "sycldnn/pooling/operators.h"
#include "sycldnn/pooling/params.h"
#include "sycldnn/pooling/sizes.h"

#include "sycldnn/reduce/launch.h"
#include "sycldnn/reduce/params.h"

#include "sycldnn/internal/pooling/launch_internal.h"

#include "sycldnn/backend/snn_backend.h"
#include "sycldnn/backend/snn_usm_backend.h"
#include "sycldnn/reduce/launch.h"

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
 * \tparam Direction  Forward or Backprop
 * \param [in] params User provided parameters to validate
 * \return An SNNStatus object containing either \ref StatusCode::OK if all
 *         parameters are valid, or \ref StatusCode::InvalidParameter otherwise.
 */
template <typename Direction>
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
  SNN_VALIDATE_PARAM(
      params.input_format == sycldnn::DataFormat::NHWC ||
          (params.input_format == sycldnn::DataFormat::NCHW &&
           std::is_same<Direction, Forward>::value),
      "Currently SYCL-DNN pooling supports the NHWC and NCHW data formats.");
  return StatusCode::OK;
}

/**
 * Checks if pooling parameters are compatible with launching a reduction
 * instead(is it a global pooling)
 * \param pp Struct containing pooling parameters
 * \return true if can be a reduction
 * \return false reduction cannot replace pooling
 */
bool is_full_reduction(const PoolingParams& pp) {
  bool equal_rows = (pp.in_rows == pp.window_rows);
  bool equal_cols = (pp.in_cols == pp.window_cols);
  bool stride_is_one = (pp.stride_rows == 1 && pp.stride_cols == 1);
  bool no_padding = (pp.pad_rows == 0 && pp.pad_cols == 0);

  return (equal_cols && equal_rows && stride_is_one && no_padding);
}

/**
 * Converts pooling parameters to ND reduce parameters to perform a global
 * pooling operation.
 * \param pp Struct containing pooling parameters
 * \return Struct containing NDReduce parameters
 */
sycldnn::reduce::NDReduceParams pp_to_reduce(const PoolingParams& pp) {
  sycldnn::reduce::NDReduceParams params;
  if (pp.input_format == sycldnn::DataFormat::NHWC) {
    params.dims = {pp.batch, pp.in_rows, pp.in_cols, pp.channels};
    params.axes = {1, 2};
  } else if (pp.input_format == sycldnn::DataFormat::NCHW) {
    params.dims = {pp.batch, pp.channels, pp.in_rows, pp.in_cols};
    params.axes = {2, 3};
  }
  params.prepare_output = true;
  return params;
}

/** Stuct to launch reduction is case of global pooling, specialises on the
 * operation and backend */
template <template <typename> class PoolType, typename T, typename Backend>
struct Reduce;

/** \copydoc  sycldnn::pooling::internal::Reduce */
template <typename T>
struct Reduce<Average, T, backend::SNNBackend> {
  /**
   * Launches reduction is case of global pooling, specialises on the operation
   * and backend
   *
   * @param input input memory
   * @param output output memory
   * @param params Reduction parameters
   * @param backend backend for reductions
   * @param events dependecies
   * @return SNNStatus
   */
  static SNNStatus launch(
      typename backend::SNNBackend::template pointer_type<T const> input,
      typename backend::SNNBackend::template pointer_type<T> output,
      sycldnn::reduce::NDReduceParams const& params,
      backend::SNNBackend& backend,
      const std::vector<cl::sycl::event>& events = {}) {
    SNN_UNUSED_VAR(events);
    return sycldnn::reduce::launch<float, sycldnn::reduce::Mean>(
        input, output, params, backend, {}, 0);
  }
};

/** \copydoc  sycldnn::pooling::internal::Reduce */
template <typename T>
struct Reduce<Average, T, backend::SNNUSMBackend> {
  /**
   * Launches reduction is case of global pooling, specialises on the operation
   * and backend
   *
   * @param input input memory
   * @param output output memory
   * @param params Reduction parameters
   * @param backend backend for reductions
   * @param events dependecies
   * @return SNNStatus
   */
  static SNNStatus launch(
      typename backend::SNNUSMBackend::template pointer_type<T const> input,
      typename backend::SNNUSMBackend::template pointer_type<T> output,
      sycldnn::reduce::NDReduceParams const& params,
      backend::SNNUSMBackend& backend,
      const std::vector<cl::sycl::event>& events = {}) {
    return sycldnn::reduce::launch<float, sycldnn::reduce::Mean>(
        input, output, params, backend, {}, 0, events);
  }
};

/** \copydoc  sycldnn::pooling::internal::Reduce */
template <typename T>
struct Reduce<Max, T, backend::SNNBackend> {
  /**
   * Launches reduction is case of global pooling, specialises on the operation
   * and backend
   *
   * @param input input memory
   * @param output output memory
   * @param params Reduction parameters
   * @param backend backend for reductions
   * @param events dependecies
   * @return SNNStatus
   */
  static SNNStatus launch(
      typename backend::SNNBackend::template pointer_type<T const> input,
      typename backend::SNNBackend::template pointer_type<T> output,
      sycldnn::reduce::NDReduceParams const& params,
      backend::SNNBackend& backend,
      const std::vector<cl::sycl::event>& events = {}) {
    SNN_UNUSED_VAR(events);
    return sycldnn::reduce::launch<float, sycldnn::reduce::Max>(
        input, output, params, backend, {}, 0);
  }
};

/** \copydoc  sycldnn::pooling::internal::Reduce */
template <typename T>
struct Reduce<Max, T, backend::SNNUSMBackend> {
  /**
   * Launches reduction is case of global pooling, specialises on the operation
   * and backend
   *
   * @param input input memory
   * @param output output memory
   * @param params Reduction parameters
   * @param backend backend for reductions
   * @param events dependecies
   * @return SNNStatus
   */
  static SNNStatus launch(
      typename backend::SNNUSMBackend::template pointer_type<T const> input,
      typename backend::SNNUSMBackend::template pointer_type<T> output,
      sycldnn::reduce::NDReduceParams const& params,
      backend::SNNUSMBackend& backend,
      const std::vector<cl::sycl::event>& events = {}) {
    return sycldnn::reduce::launch<float, sycldnn::reduce::Max>(
        input, output, params, backend, {}, 0, events);
  }
};

/** \copydoc  sycldnn::pooling::internal::Reduce */
template <typename T>
struct Reduce<MaxWithNan, T, backend::SNNBackend> {
  /**
   * Launches reduction is case of global pooling, specialises on the operation
   * and backend
   *
   * @param input input memory
   * @param output output memory
   * @param params Reduction parameters
   * @param backend backend for reductions
   * @param events dependecies
   * @return SNNStatus
   */
  static SNNStatus launch(
      typename backend::SNNBackend::template pointer_type<T const> input,
      typename backend::SNNBackend::template pointer_type<T> output,
      sycldnn::reduce::NDReduceParams const& params,
      backend::SNNBackend& backend,
      const std::vector<cl::sycl::event>& events = {}) {
    SNN_UNUSED_VAR(events);
    return sycldnn::reduce::launch<float, sycldnn::reduce::Max>(
        input, output, params, backend, {}, 0);
  }
};

/** \copydoc  sycldnn::pooling::internal::Reduce */
template <typename T>
struct Reduce<MaxWithNan, T, backend::SNNUSMBackend> {
  /**
   * Launches reduction is case of global pooling, specialises on the operation
   * and backend
   *
   * @param input input memory
   * @param output output memory
   * @param params Reduction parameters
   * @param backend backend for reductions
   * @param events dependecies
   * @return SNNStatus
   */
  static SNNStatus launch(
      typename backend::SNNUSMBackend::template pointer_type<T const> input,
      typename backend::SNNUSMBackend::template pointer_type<T> output,
      sycldnn::reduce::NDReduceParams const& params,
      backend::SNNUSMBackend& backend,
      const std::vector<cl::sycl::event>& events = {}) {
    return sycldnn::reduce::launch<float, sycldnn::reduce::Max>(
        input, output, params, backend, {}, 0, events);
  }
};

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
          typename = typename std::enable_if<
              sycldnn::backend::is_buffer_backend_v<Backend> &&
                  !internal::IsMaxGradient<T, PoolType, Direction>::value,
              int>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T> output,
                 const PoolingParams& pp, Backend& backend) {
  auto validation_status = internal::validate_params<Direction>(pp);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }
  if (std::is_same_v<Direction, pooling::Forward> &&
      internal::is_full_reduction(pp)) {
    auto reduce_params = internal::pp_to_reduce(pp);

    return internal::Reduce<PoolType, T, Backend>::launch(
        input, output, reduce_params, backend, {});
  } else {
    return internal::sublaunch<T, PoolType, Direction, Backend>(
        input, output, pp, backend, {});
  }
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
          typename = typename std::enable_if<
              sycldnn::backend::is_buffer_backend_v<Backend> &&
                  internal::IsMaxGradient<T, PoolType, Direction>::value,
              int>::type>
SNNStatus launch(
    typename Backend::template pointer_type<T const> input_data,
    typename Backend::template pointer_type<T const> output_data,
    typename Backend::template pointer_type<T const> input_backprop,
    typename Backend::template pointer_type<T> output, const PoolingParams& pp,
    Backend& backend) {
  auto validation_status = internal::validate_params<Direction>(pp);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }

  return internal::sublaunch<T, PoolType, Direction, Backend>(
      input_data, output_data, input_backprop, output, pp, backend, {});
}

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
 * \param [in] events    USM dependency events.
 * \return An \ref SNNStatus containing the SYCL event tied to the kernel
 *         launches and a \ref StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, template <typename> class PoolType, typename Direction,
          typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_usm_backend_v<Backend> &&
                  !internal::IsMaxGradient<T, PoolType, Direction>::value,
              int>::type>
SNNStatus launch(typename Backend::template pointer_type<T const> input,
                 typename Backend::template pointer_type<T> output,
                 const PoolingParams& pp, Backend& backend,
                 const std::vector<cl::sycl::event>& events = {}) {
  auto validation_status = internal::validate_params<Direction>(pp);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }
  if (std::is_same_v<Direction, pooling::Forward> &&
      internal::is_full_reduction(pp)) {
    auto reduce_params = internal::pp_to_reduce(pp);
    return internal::Reduce<PoolType, T, Backend>::launch(
        input, output, reduce_params, backend, events);
  } else {
    return internal::sublaunch<T, PoolType, Direction, Backend>(
        input, output, pp, backend, events);
  }
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
 * \param [in] events          USM dependency events.
 * \return An \ref SNNStatus containing the SYCL event tied to the kernel
 *         launches and a \ref StatusCode enum showing if the launch was OK or
 *         whether it encountered some problem.
 */
template <typename T, template <typename> class PoolType, typename Direction,
          typename Backend,
          typename = typename std::enable_if<
              sycldnn::backend::is_usm_backend_v<Backend> &&
                  internal::IsMaxGradient<T, PoolType, Direction>::value,
              int>::type>
SNNStatus launch(
    typename Backend::template pointer_type<T const> input_data,
    typename Backend::template pointer_type<T const> output_data,
    typename Backend::template pointer_type<T const> input_backprop,
    typename Backend::template pointer_type<T> output, const PoolingParams& pp,
    Backend& backend, const std::vector<cl::sycl::event>& events = {}) {
  auto validation_status = internal::validate_params<Direction>(pp);
  if (validation_status.status != StatusCode::OK) {
    return validation_status;
  }

  return internal::sublaunch<T, PoolType, Direction, Backend>(
      input_data, output_data, input_backprop, output, pp, backend, events);
}

}  // namespace pooling
}  // namespace sycldnn

#endif  // SYCLDNN_INCLUDE_POOLING_LAUNCH_H_
