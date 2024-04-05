#include "bench/common/cudnn_common.h"

namespace sycldnn {
namespace bench {

enum cudnnConvolutionAlgo_t {
  IMPLICIT_GEMM_NONDETERMINISTIC = 0,
  IMPLICIT_GEMM_DETERMINISTIC,
  IMPLICIT_GEMM_PRECOMP,
  GEMM,
  WINOGRAD,
  WINOGRAD_NONFUSED,
  FFT
};

std::string get_cudnn_conv_algo_str(cudnnConvolutionAlgo_t algo) {
  switch (algo) {
    case cudnnConvolutionAlgo_t::IMPLICIT_GEMM_NONDETERMINISTIC:
      return "IMPLICIT_GEMM_NONDETERMINISTIC";
    case cudnnConvolutionAlgo_t::IMPLICIT_GEMM_DETERMINISTIC:
      return "IMPLICIT_GEMM_DETERMINISTIC";
    case cudnnConvolutionAlgo_t::IMPLICIT_GEMM_PRECOMP:
      return "IMPLICIT_PRECOMP_GEMM";
    case cudnnConvolutionAlgo_t::GEMM:
      return "GEMM";
    case cudnnConvolutionAlgo_t::WINOGRAD:
      return "WINOGRAD";
    case cudnnConvolutionAlgo_t::WINOGRAD_NONFUSED:
      return "WINOGRAD_NONFUSED";
    case cudnnConvolutionAlgo_t::FFT:
      return "FFT";
    default:
      throw std::runtime_error("Unsupported CuDNN algorithm");
  }
}

// Forward CuDNN convolution algorithms
template <cudnnConvolutionAlgo_t Algo>
struct get_cudnn_fwd_conv_algo;
template <>
struct get_cudnn_fwd_conv_algo<
    cudnnConvolutionAlgo_t::IMPLICIT_GEMM_DETERMINISTIC> {
  static constexpr auto value = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
};
template <>
struct get_cudnn_fwd_conv_algo<cudnnConvolutionAlgo_t::GEMM> {
  static constexpr auto value = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
};
template <>
struct get_cudnn_fwd_conv_algo<cudnnConvolutionAlgo_t::WINOGRAD> {
  static constexpr auto value = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
};
template <>
struct get_cudnn_fwd_conv_algo<cudnnConvolutionAlgo_t::WINOGRAD_NONFUSED> {
  static constexpr auto value = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
};
template <>
struct get_cudnn_fwd_conv_algo<cudnnConvolutionAlgo_t::FFT> {
  static constexpr auto value = CUDNN_CONVOLUTION_FWD_ALGO_FFT;
};

// Backward input CuDNN convolution algorithms
template <cudnnConvolutionAlgo_t Algo>
struct get_cudnn_bwd_data_conv_algo;
template <>
struct get_cudnn_bwd_data_conv_algo<
    cudnnConvolutionAlgo_t::IMPLICIT_GEMM_DETERMINISTIC> {
  static constexpr auto value = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
};
template <>
struct get_cudnn_bwd_data_conv_algo<
    cudnnConvolutionAlgo_t::IMPLICIT_GEMM_NONDETERMINISTIC> {
  static constexpr auto value = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
};
template <>
struct get_cudnn_bwd_data_conv_algo<cudnnConvolutionAlgo_t::WINOGRAD> {
  static constexpr auto value = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
};
template <>
struct get_cudnn_bwd_data_conv_algo<cudnnConvolutionAlgo_t::WINOGRAD_NONFUSED> {
  static constexpr auto value =
      CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
};
template <>
struct get_cudnn_bwd_data_conv_algo<cudnnConvolutionAlgo_t::FFT> {
  static constexpr auto value = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT;
};

// Backward filter CuDNN convolution algorithms
template <cudnnConvolutionAlgo_t Algo>
struct get_cudnn_bwd_filter_conv_algo;
template <>
struct get_cudnn_bwd_filter_conv_algo<
    cudnnConvolutionAlgo_t::IMPLICIT_GEMM_DETERMINISTIC> {
  static constexpr auto value = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
};
template <>
struct get_cudnn_bwd_filter_conv_algo<
    cudnnConvolutionAlgo_t::IMPLICIT_GEMM_NONDETERMINISTIC> {
  static constexpr auto value = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
};
template <>
struct get_cudnn_bwd_filter_conv_algo<
    cudnnConvolutionAlgo_t::IMPLICIT_GEMM_PRECOMP> {
  static constexpr auto value = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3;
};
template <>
struct get_cudnn_bwd_filter_conv_algo<cudnnConvolutionAlgo_t::WINOGRAD> {
  static constexpr auto value = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD;
};
template <>
struct get_cudnn_bwd_filter_conv_algo<
    cudnnConvolutionAlgo_t::WINOGRAD_NONFUSED> {
  static constexpr auto value =
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;
};
template <>
struct get_cudnn_bwd_filter_conv_algo<cudnnConvolutionAlgo_t::FFT> {
  static constexpr auto value = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT;
};
}  // namespace bench
}  // namespace sycldnn
