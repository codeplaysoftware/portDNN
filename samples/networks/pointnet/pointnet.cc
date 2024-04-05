#if defined(SNN_TEST_PORTBLAS)
#include <portblas.h>
#else
#include "sycldnn/backend/snn_usm_backend.h"
#include "sycldnn/matmul/launch.h"
#include "sycldnn/matmul/params.h"
#endif

#include "sycldnn/compat/activation.hpp"
#include "sycldnn/compat/batchnorm.hpp"
#include "sycldnn/compat/convolution.hpp"
#include "sycldnn/compat/pooling.hpp"
#include "sycldnn/compat/softmax.hpp"

#include "tools/io_helpers.h"

#define DEBUG_LAYERS 1

using namespace sycldnn::compat;
using DType = float;
static std::array<std::string, 10> object_map = {
    "bathtub", "bed",         "chair", "desk",  "dresser",
    "monitor", "night stand", "sofa",  "table", "toilet"};

namespace helpers {

template <typename T>
void copy_to_device(std::vector<char>& inputs, T* dev_ptr, cl::sycl::queue q) {
  q.submit([&](cl::sycl::handler& cgh) {
     cgh.copy(inputs.data(), reinterpret_cast<char*>(dev_ptr), inputs.size());
   }).wait_and_throw();
}

void dump_desc(const TensorDescriptor& desc) {
  int n, c, h, w;
  int s_n, s_c, s_h, s_w;
  SNNDataType data_type;
  getTensor4dDescriptor(desc, &data_type, &n, &c, &h, &w, &s_n, &s_c, &s_h,
                        &s_w);
  std::cout << "\tdims: [" << n << "," << c << "," << h << "," << w << "]\n";
}

void dump_desc(const FilterDescriptor& desc) {
  int k, c, h, w;
  sycldnn::FilterFormat format;
  SNNDataType data_type;
  getFilter4dDescriptor(desc, &data_type, &format, &k, &c, &h, &w);
  std::cout << "\tdims: [" << k << "," << c << "," << h << "," << w << "]\n";
}

void dump_desc(const ConvolutionDescriptor& desc) {
  std::cout << "\tpad: [" << desc.getPadH() << "," << desc.getPadW() << "]\n";
  std::cout << "\tstride: [" << desc.getStrideH() << "," << desc.getStrideW()
            << "]\n";
}

void dump_desc(const PoolingDescriptor& desc) {
  std::cout << "\tmode: " << static_cast<int>(desc.getMode()) << "\n";
  std::cout << "\twindow: [" << desc.getWindowH() << "," << desc.getWindowW()
            << "]\n";
  std::cout << "\tpad: [" << desc.getPadH() << "," << desc.getPadW() << "]\n";
  std::cout << "\tstride: [" << desc.getStrideH() << "," << desc.getStrideW()
            << "]\n";
}
}  // namespace helpers

template <typename T>
struct Layer {
  explicit Layer(SNNHandle& handle)
      : handle_(handle), out_ptr_(nullptr), out_desc_({}) {}

  Layer(SNNHandle& handle, TensorDescriptor const& out_desc)
      : handle_(handle), out_ptr_(nullptr), out_desc_(out_desc) {}

  virtual ~Layer() { sycl::free(out_ptr_, handle_.getQueue()); }

  virtual sycldnn::SNNStatus execute(T* in_ptr) = 0;

  T* get_output_ptr() { return out_ptr_; };

  TensorDescriptor& get_output_desc() { return out_desc_; }

  void dump_out_desc() {
    std::cout << "Output:\n";
    helpers::dump_desc(out_desc_);
  }

  virtual void dump_descs() = 0;

  SNNHandle& handle_;

 protected:
  T* out_ptr_ = nullptr;
  TensorDescriptor out_desc_;
};

template <typename T>
struct ConvLayer : public Layer<T> {
  ConvLayer(SNNHandle& handle, std::string const& filter_file, const int in_n,
            const int in_c, const int in_h, const int in_w, const int filt_f,
            const int filt_c, const int filt_h, const int filt_w,
            sycldnn::DataFormat format = sycldnn::DataFormat::NHWC,
            SNNDataType data_type = SNNDataType::SNN_FLOAT)
      : Layer<T>(handle) {
    auto q = handle.getQueue();

    // Setup input tensor descriptor
    setTensor4dDescriptor(in_desc_, format, data_type, in_n, in_c, in_h, in_w);

    // Setup filter descriptor
    setFilter4dDescriptor(filt_desc_, data_type, format, filt_f, filt_c, filt_h,
                          filt_w);

    // Allocate filter memory
    filt_ptr_ = static_cast<T*>(cl::sycl::malloc_device(
        filt_h * filt_w * filt_c * filt_f * sizeof(T), q));
    // Read weights from binary file
    auto weights = sycldnn::helpers::read_binary_data(filter_file);
    // Copy weights to device
    helpers::copy_to_device<T>(weights, filt_ptr_, q);

    // Setup convolution descriptor
    setConvolution2dDescriptor(conv_desc_, 0, 0, 1, 1, 1, 1,
                               ConvolutionMode::CROSS_CORRELATION);

    // Get output dimensions
    int out_n, out_c, out_h, out_w;
    getConvolution2dForwardOutputDim(conv_desc_, in_desc_, filt_desc_, &out_n,
                                     &out_c, &out_h, &out_w);

    // Setup output tensor descriptor
    setTensor4dDescriptor(this->out_desc_, format, data_type, out_n, out_c,
                          out_h, out_w);
    this->out_ptr_ = static_cast<T*>(
        cl::sycl::malloc_device(out_n * out_h * out_c * out_w * sizeof(T), q));

    // Find optimal algorithm
    convolutionFwdAlgoPerf_t perf_results;
    int req_alg_count = 1;
    int ret_alg_count;
    findConvolutionForwardAlgorithm(this->handle_, in_desc_, filt_desc_,
                                    conv_desc_, this->out_desc_, req_alg_count,
                                    &ret_alg_count, &perf_results);
    algo_ = perf_results.algo[0];

    ws_size_ = 0;
    getConvolutionForwardWorkspaceSize(this->handle_, in_desc_, filt_desc_,
                                       conv_desc_, this->out_desc_, algo_,
                                       &ws_size_);
    if (ws_size_) {
      // Allocate workspace memory if needed
      ws_ptr_ =
          static_cast<T*>(cl::sycl::malloc_device(ws_size_ * sizeof(T), q));
    }
  }
  sycldnn::SNNStatus execute(T* in_ptr) override {
    return convolutionForward(this->handle_, &alpha_, in_desc_, in_ptr,
                              filt_desc_, filt_ptr_, conv_desc_, algo_, ws_ptr_,
                              ws_size_, &beta_, this->out_desc_,
                              this->out_ptr_);
  }

  ~ConvLayer() {
    if (ws_ptr_) {
      sycl::free(ws_ptr_, this->handle_.getQueue());
    }
    sycl::free(filt_ptr_, this->handle_.getQueue());
  }

  virtual void dump_descs() override {
    std::cout << "Layer type: Convolution\n";
    std::cout << "Conv:\n";
    helpers::dump_desc(conv_desc_);
    std::cout << "Input:\n";
    helpers::dump_desc(in_desc_);
    std::cout << "Filter:\n";
    helpers::dump_desc(filt_desc_);
  }

 private:
  T* ws_ptr_ = nullptr;
  T* filt_ptr_ = nullptr;
  size_t ws_size_;
  sycldnn::conv2d::Algorithm algo_;
  TensorDescriptor in_desc_;
  FilterDescriptor filt_desc_;
  ConvolutionDescriptor conv_desc_;
  float alpha_ = 1.0f;
  float beta_ = 0.0f;
};

template <typename T>
struct BiasAddLayer : public Layer<T> {
  BiasAddLayer(SNNHandle& handle, std::string const& bias_file, const int batch,
               const int channels, const int rows, const int cols,
               sycldnn::DataFormat format = sycldnn::DataFormat::NHWC,
               SNNDataType data_type = SNNDataType::SNN_FLOAT)
      : Layer<T>(handle) {
    setTensor4dDescriptor(in_desc_, format, data_type, batch, channels, rows,
                          cols);
    setTensor4dDescriptor(this->out_desc_, format, data_type, batch, channels,
                          rows, cols);
    setTensor4dDescriptor(bias_desc_, format, data_type, 1, channels, 1, 1);

    auto q = handle.getQueue();

    bias_ptr_ =
        static_cast<T*>(cl::sycl::malloc_device(channels * sizeof(T), q));
    // Read weights from binary file
    auto weights = sycldnn::helpers::read_binary_data(bias_file);
    // Copy weights to device
    helpers::copy_to_device<T>(weights, bias_ptr_, q);

    this->out_ptr_ = static_cast<T*>(
        cl::sycl::malloc_device(batch * channels * rows * cols * sizeof(T), q));
  }

  sycldnn::SNNStatus execute(T* in_ptr) override {
    this->handle_.getQueue().memcpy(this->out_ptr_, in_ptr,
                                    this->out_desc_.getSize() * sizeof(T));
    return addTensor(this->handle_, &alpha_, bias_desc_, bias_ptr_, &beta_,
                     this->out_desc_, this->out_ptr_);
  }
  ~BiasAddLayer() = default;

  virtual void dump_descs() override {
    std::cout << "Layer type: BiasAdd\n";
    std::cout << "Bias:\n";
    helpers::dump_desc(bias_desc_);
    std::cout << "In:\n";
    helpers::dump_desc(this->out_desc_);
  }

 private:
  TensorDescriptor in_desc_;
  TensorDescriptor bias_desc_;

  T* bias_ptr_ = nullptr;

  float alpha_ = 1.0f;
  float beta_ = 0.0f;
};

template <typename T>
struct BatchNormLayer : public Layer<T> {
  BatchNormLayer(SNNHandle& handle, std::string const& scale_file,
                 std::string const& bias_file, std::string const& mean_file,
                 std::string const& var_file, const int batch,
                 const int channels, const int rows, const int cols,
                 BatchNormMode mode = BatchNormMode::BATCHNORM_SPATIAL,
                 sycldnn::DataFormat format = sycldnn::DataFormat::NHWC,
                 SNNDataType data_type = SNNDataType::SNN_FLOAT)
      : Layer<T>(handle), mode_{mode} {
    setTensor4dDescriptor(in_desc_, format, data_type, batch, channels, rows,
                          cols);
    setTensor4dDescriptor(this->out_desc_, format, data_type, batch, channels,
                          rows, cols);
    setTensor4dDescriptor(param_desc_, format, data_type, 1, channels, 1, 1);

    auto q = handle.getQueue();
    this->out_ptr_ = static_cast<T*>(
        cl::sycl::malloc_device(batch * channels * rows * cols * sizeof(T), q));

    auto scale = sycldnn::helpers::read_binary_data(scale_file);
    scale_ptr_ =
        static_cast<T*>(cl::sycl::malloc_device(channels * sizeof(T), q));
    helpers::copy_to_device<T>(scale, scale_ptr_, q);

    auto bias = sycldnn::helpers::read_binary_data(bias_file);
    bias_ptr_ =
        static_cast<T*>(cl::sycl::malloc_device(channels * sizeof(T), q));
    helpers::copy_to_device<T>(bias, bias_ptr_, q);

    auto mean = sycldnn::helpers::read_binary_data(mean_file);
    mean_ptr_ =
        static_cast<T*>(cl::sycl::malloc_device(channels * sizeof(T), q));
    helpers::copy_to_device<T>(mean, mean_ptr_, q);

    auto var = sycldnn::helpers::read_binary_data(var_file);
    var_ptr_ =
        static_cast<T*>(cl::sycl::malloc_device(channels * sizeof(T), q));
    helpers::copy_to_device<T>(var, var_ptr_, q);
  }

  sycldnn::SNNStatus execute(T* in_ptr) override {
    return batchNormalizationForwardInference(
        this->handle_, mode_, &alpha_, &beta_, in_desc_, in_ptr,
        this->out_desc_, this->out_ptr_, param_desc_, scale_ptr_, bias_ptr_,
        mean_ptr_, var_ptr_, eps_);
  }

  ~BatchNormLayer() {
    sycl::free(scale_ptr_, this->handle_.getQueue());
    sycl::free(bias_ptr_, this->handle_.getQueue());
    sycl::free(mean_ptr_, this->handle_.getQueue());
    sycl::free(var_ptr_, this->handle_.getQueue());
  }

  virtual void dump_descs() override {
    std::cout << "Layer type: BatchNorm\n";
    std::cout << "Param:\n";
    helpers::dump_desc(param_desc_);
    std::cout << "In:\n";
    helpers::dump_desc(this->in_desc_);
  }

 private:
  BatchNormMode mode_;

  TensorDescriptor in_desc_;
  TensorDescriptor param_desc_;

  T* scale_ptr_ = nullptr;
  T* bias_ptr_ = nullptr;
  T* mean_ptr_ = nullptr;
  T* var_ptr_ = nullptr;

  float eps_ = 1e-5;
  float alpha_ = 1.0;
  float beta_ = 0.0;
};

template <typename T>
struct ActivationLayer : public Layer<T> {
  ActivationLayer(SNNHandle& handle, const int batch, const int channels,
                  const int rows, const int cols,
                  ActivationMode mode = ActivationMode::ACTIVATION_RELU,
                  NanPropagation nan_prop = NanPropagation::NOT_PROPAGATE_NAN,
                  sycldnn::DataFormat format = sycldnn::DataFormat::NHWC,
                  SNNDataType data_type = SNNDataType::SNN_FLOAT)
      : Layer<T>(handle) {
    setTensor4dDescriptor(in_desc_, format, data_type, batch, channels, rows,
                          cols);

    setTensor4dDescriptor(this->out_desc_, format, data_type, batch, channels,
                          rows, cols);
    auto q = handle.getQueue();
    this->out_ptr_ = static_cast<T*>(
        cl::sycl::malloc_device(batch * channels * rows * cols * sizeof(T), q));

    setActivationDescriptor(act_desc_, mode, nan_prop, coef_);
  }

  sycldnn::SNNStatus execute(T* in_ptr) override {
    return activationForward(this->handle_, act_desc_, &alpha_, in_desc_,
                             in_ptr, &beta_, this->out_desc_, this->out_ptr_);
  }
  ~ActivationLayer() = default;

  virtual void dump_descs() override {
    std::cout << "Layer type: Activation\n";
    std::cout << "In:\n";
    helpers::dump_desc(in_desc_);
  }

 private:
  TensorDescriptor in_desc_;

  ActivationDescriptor act_desc_;

  float coef_ = 0.f;
  float alpha_ = 1.f;
  float beta_ = 0.f;
};

template <typename T>
struct GlobalMaxPoolLayer : public Layer<T> {
  GlobalMaxPoolLayer(
      SNNHandle& handle, const int batch, const int channels, const int rows,
      const int cols,
      NanPropagation nan_prop = NanPropagation::NOT_PROPAGATE_NAN,
      sycldnn::DataFormat format = sycldnn::DataFormat::NHWC,
      SNNDataType data_type = SNNDataType::SNN_FLOAT)
      : Layer<T>(handle) {
    setTensor4dDescriptor(in_desc_, format, data_type, batch, channels, rows,
                          cols);
    setTensor4dDescriptor(this->out_desc_, format, data_type, batch, channels,
                          1, 1);
    setPooling2dDescriptor(pool_desc_, PoolingMode::POOLING_MAX_DETERMINISTIC,
                           nan_prop, rows, cols, 0, 0, rows, cols);
    auto q = this->handle_.getQueue();
    this->out_ptr_ = static_cast<T*>(
        cl::sycl::malloc_device(batch * channels * sizeof(T), q));
  }

  sycldnn::SNNStatus execute(T* in_ptr) override {
    return poolingForward(this->handle_, pool_desc_, &alpha_, in_desc_, in_ptr,
                          &beta_, this->out_desc_, this->out_ptr_);
  }
  ~GlobalMaxPoolLayer() = default;

  virtual void dump_descs() override {
    std::cout << "Layer type: GlobalMaxPool\n";
    std::cout << "Pool:\n";
    helpers::dump_desc(pool_desc_);
    std::cout << "In:\n";
    helpers::dump_desc(in_desc_);
  }

 private:
  TensorDescriptor in_desc_;
  PoolingDescriptor pool_desc_;

  float alpha_ = 1.0f;
  float beta_ = 0.0f;
};

template <typename T>
struct FCLayer : public Layer<T> {
  FCLayer(SNNHandle& handle, std::string const& weights_file, const int batch,
          const int in_channels, const int out_channels,
          SNNDataType data_type = SNNDataType::SNN_FLOAT)
      : Layer<T>(handle), m_{batch}, k_{in_channels}, n_{out_channels} {
    auto q = this->handle_.getQueue();

    auto weights = sycldnn::helpers::read_binary_data(weights_file);
    weights_ptr_ = static_cast<T*>(
        cl::sycl::malloc_device(in_channels * out_channels * sizeof(T), q));
    helpers::copy_to_device<T>(weights, weights_ptr_, q);

    this->out_ptr_ = static_cast<T*>(
        cl::sycl::malloc_device(batch * out_channels * sizeof(T), q));
    // Setting up tensor descriptor because it's needed for the output size
    // calulation. DataFormat is hardcoded because it has no real effect here
    setTensor4dDescriptor(this->out_desc_, sycldnn::DataFormat::NHWC, data_type,
                          batch, out_channels, 1, 1);
  }

  sycldnn::SNNStatus execute(T* in_ptr) override {
#if defined(SNN_TEST_PORTBLAS)
    // PortBLAS is column-major vs sycl-dnn row-major so we swap lhs and rhs.
    blas::SB_Handle sb_handle{this->handle_.getQueue()};
    auto event =
        blas::_gemm(sb_handle, 'n', 'n', n_, m_, k_, alpha_, weights_ptr_, n_,
                    in_ptr, k_, beta_, this->out_ptr_, n_)
            .back();
    return {event, sycldnn::StatusCode::OK};
#else
    sycldnn::matmul::MatmulParams params{1, m_, k_, n_, T{0}};
    // Fallsback to sycl-dnn matmul if portblas is inaccessible.
    return sycldnn::matmul::launch<T, false, false,
                                   sycldnn::backend::SNNUSMBackend>(
        in_ptr, weights_ptr_, this->out_ptr_, params,
        this->handle_.getBackend());
#endif
  }

  ~FCLayer() { sycl::free(weights_ptr_, this->handle_.getQueue()); }

  virtual void dump_descs() override { std::cout << "Layer type: FC\n"; }

 private:
  int m_, k_, n_;

  T* weights_ptr_ = nullptr;

  float alpha_ = 1.0f;
  float beta_ = 0.0f;
};

template <typename T>
struct MMLayer : public Layer<T> {
  MMLayer(SNNHandle& handle, T* lhs_ptr, const int batch, const int m,
          const int k, const int n,
          SNNDataType data_type = SNNDataType::SNN_FLOAT)
      : Layer<T>(handle),
        lhs_ptr_{lhs_ptr},
        batch_{batch},
        m_{m},
        k_{k},
        n_{n} {
    auto q = this->handle_.getQueue();

    this->out_ptr_ =
        static_cast<T*>(cl::sycl::malloc_device(batch * m * n * sizeof(T), q));
    // Setting up tensor descriptor because it's needed for the output size
    // calulation. DataFormat is hardcoded because it has no real effect here
    setTensor4dDescriptor(this->out_desc_, sycldnn::DataFormat::NHWC, data_type,
                          batch, m, n, 1);
  }

  sycldnn::SNNStatus execute(T* rhs_ptr) override {
#if defined(SNN_TEST_PORTBLAS) || defined(SNN_BENCH_PORTBLAS)
    // PortBLAS is column-major vs sycl-dnn row-major so we swap lhs and rhs.
    blas::SB_Handle sb_handle{this->handle_.getQueue()};
    auto event =
        blas::_gemm_batched(sb_handle, 'n', 'n', n_, m_, k_, alpha_, rhs_ptr,
                            n_, lhs_ptr_, k_, beta_, this->out_ptr_, n_, batch_)
            .back();
    return {event, sycldnn::StatusCode::OK};
#else
    sycldnn::matmul::MatmulParams params{batch_, m_, k_, n_, T{0}};
    // Fallsback to sycl-dnn matmul if portblas is inaccessible.
    return sycldnn::matmul::launch<T, false, false,
                                   sycldnn::backend::SNNUSMBackend>(
        lhs_ptr_, rhs_ptr, this->out_ptr_, params, this->handle_.getBackend());
#endif
  }

  ~MMLayer() = default;

  virtual void dump_descs() override { std::cout << "Layer type: Matmul\n"; }

 private:
  T* lhs_ptr_ = nullptr;

  int batch_, m_, k_, n_;

  float alpha_ = 1.0f;
  float beta_ = 0.0f;
};

template <typename T>
struct SoftmaxLayer : public Layer<T> {
  SoftmaxLayer(SNNHandle& handle, const int batch, const int channels,
               const int rows, const int cols,
               SoftmaxAlgorithm algo = SoftmaxAlgorithm::SOFTMAX_ACCURATE,
               SoftmaxMode mode = SoftmaxMode::SOFTMAX_MODE_CHANNEL,
               sycldnn::DataFormat format = sycldnn::DataFormat::NHWC,
               SNNDataType data_type = SNNDataType::SNN_FLOAT)
      : Layer<T>(handle), algo_{algo}, mode_{mode} {
    setTensor4dDescriptor(in_desc_, format, data_type, batch, channels, rows,
                          cols);
    setTensor4dDescriptor(this->out_desc_, format, data_type, batch, channels,
                          rows, cols);

    this->out_ptr_ = static_cast<T*>(cl::sycl::malloc_device(
        batch * channels * rows * cols * sizeof(T), this->handle_.getQueue()));
  }

  sycldnn::SNNStatus execute(T* in_ptr) override {
    return softmaxForward(this->handle_, algo_, mode_, &alpha_, in_desc_,
                          in_ptr, &beta_, this->out_desc_, this->out_ptr_);
  }
  ~SoftmaxLayer() = default;

  virtual void dump_descs() override {
    std::cout << "Layer type: Softmax\n";
    std::cout << "In:\n";
    helpers::dump_desc(in_desc_);
  }

 private:
  SoftmaxAlgorithm algo_;

  SoftmaxMode mode_;

  TensorDescriptor in_desc_;

  float alpha_ = 1.0f;
  float beta_ = 0.0f;
};

template <typename T>
struct Network {
  void add_layer(std::unique_ptr<Layer<T>> layer) {
    layers.emplace_back(std::move(layer));
  }

  void execute(T* in_ptr) {
    for (auto& layer : layers) {
      layer->execute(in_ptr);
#if DEBUG_LAYERS
      layer->handle_.getQueue().wait_and_throw();
      layer->dump_descs();
      layer->dump_out_desc();
      std::cout << "***\n";
#endif
      in_ptr = layer->get_output_ptr();
    }
  }

  TensorDescriptor& get_last_output_desc() {
    return layers.back()->get_output_desc();
  }

  T* get_last_output_ptr() { return layers.back()->get_output_ptr(); }

  std::vector<T> get_output_as_host_vec() {
    auto& last_layer = layers.back();
    auto& output_desc = last_layer->get_output_desc();
    auto& handle = last_layer->handle_;
    std::vector<T> output(output_desc.getSize());
    handle.getQueue()
        .memcpy(output.data(), last_layer->get_output_ptr(),
                output.size() * sizeof(T))
        .wait_and_throw();

    return output;
  }

  void dump_output() {
    auto output = get_output_as_host_vec();
    std::cout << "Output:\n";
    for (auto e : output) {
      std::cout << e << ", ";
    }
    std::cout << "\n";
  }

  std::vector<std::unique_ptr<Layer<T>>> layers;
};

template <typename T>
inline void add_conv_layer(Network<T>& net, SNNHandle& handle,
                           std::string const& filter_file, const int in_n,
                           const int in_c, const int in_h, const int in_w,
                           const int filt_f, const int filt_c, const int filt_h,
                           const int filt_w) {
  net.add_layer(std::make_unique<ConvLayer<T>>(handle, filter_file, in_n, in_c,
                                               in_h, in_w, filt_f, filt_c,
                                               filt_h, filt_w));
}

template <typename T>
inline void add_bias_layer(Network<T>& net, SNNHandle& handle,
                           std::string const& bias_file, const int n,
                           const int c, const int h, const int w) {
  net.add_layer(
      std::make_unique<BiasAddLayer<T>>(handle, bias_file, n, c, h, w));
}

template <typename T>
inline void add_batchnorm_layer(Network<T>& net, SNNHandle& handle,
                                std::string const& scale_file,
                                std::string const& bias_file,
                                std::string const& mean_file,
                                std::string const& var_file, const int n,
                                const int c, const int h, const int w) {
  net.add_layer(std::make_unique<BatchNormLayer<T>>(
      handle, scale_file, bias_file, mean_file, var_file, n, c, h, w));
}

template <typename T>
inline void add_activation_layer(
    Network<T>& net, SNNHandle& handle, const int n, const int c, const int h,
    const int w, ActivationMode mode = ActivationMode::ACTIVATION_RELU) {
  net.add_layer(std::make_unique<ActivationLayer<T>>(handle, n, c, h, w, mode));
}

template <typename T>
inline void add_global_max_pool_layer(Network<T>& net, SNNHandle& handle,
                                      const int n, const int c, const int h,
                                      const int w) {
  net.add_layer(std::make_unique<GlobalMaxPoolLayer<T>>(handle, n, c, h, w));
}

template <typename T>
inline void add_fc_layer(Network<T>& net, SNNHandle& handle,
                         std::string const& weights_file, const int batch,
                         const int in_c, const int out_c) {
  net.add_layer(
      std::make_unique<FCLayer<T>>(handle, weights_file, batch, in_c, out_c));
}

template <typename T>
inline void add_mm_layer(Network<T>& net, SNNHandle& handle, T* lhs_ptr,
                         const int batch, const int m, const int k,
                         const int n) {
  net.add_layer(std::make_unique<MMLayer<T>>(handle, lhs_ptr, batch, m, k, n));
}

template <typename T>
inline void add_softmax_layer(Network<T>& net, SNNHandle& handle, const int n,
                              const int c, const int h, const int w) {
  net.add_layer(std::make_unique<SoftmaxLayer<T>>(handle, n, c, h, w));
}

template <typename T>
inline void add_conv_bias_bnorm_relu_block(
    Network<T>& net, SNNHandle& handle, std::string const& file_directory,
    std::string const& conv_filter_file, std::string const& conv_bias_file,
    std::string const& bn_scale_file, std::string const& bn_bias_file,
    std::string const& bn_mean_file, std::string const& bn_var_file,
    const int in_n, const int in_c, const int in_h, const int in_w,
    const int out_c, const int filt_h, const int filt_w, bool add_relu = true) {
  add_conv_layer(net, handle, file_directory + conv_filter_file, in_n, in_c,
                 in_h, in_w, out_c, in_c, filt_h, filt_w);
  add_bias_layer(net, handle, file_directory + conv_bias_file, in_n, out_c,
                 in_h, in_w);
  add_batchnorm_layer(net, handle, file_directory + bn_scale_file,
                      file_directory + bn_bias_file,
                      file_directory + bn_mean_file,
                      file_directory + bn_var_file, in_n, out_c, in_h, in_w);
  if (add_relu) {
    add_activation_layer(net, handle, in_n, out_c, in_h, in_w,
                         ActivationMode::ACTIVATION_RELU);
  }
}

template <typename T>
inline void add_fc_bias_bnorm_relu_block(
    Network<T>& net, SNNHandle& handle, std::string const& file_directory,
    std::string const& fc_filter_file, std::string const& fc_bias_file,
    std::string const& bn_scale_file, std::string const& bn_bias_file,
    std::string const& bn_mean_file, std::string const& bn_var_file,
    const int batch, const int in_c, const int out_c) {
  add_fc_layer(net, handle, file_directory + fc_filter_file, batch, in_c,
               out_c);
  add_bias_layer(net, handle, file_directory + fc_bias_file, batch, out_c, 1,
                 1);
  add_batchnorm_layer(net, handle, file_directory + bn_scale_file,
                      file_directory + bn_bias_file,
                      file_directory + bn_mean_file,
                      file_directory + bn_var_file, batch, out_c, 1, 1);
  add_activation_layer(net, handle, batch, out_c, 1, 1,
                       ActivationMode::ACTIVATION_RELU);
}

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cout << "USAGE: " << argv[0] << " <directory> <image>" << std::endl;
    return 1;
  }

  SNNHandle handle;
  SNNCreate(handle);
  auto q = handle.getQueue();

  std::string data_dir{argv[1]};
  data_dir += "/";

  DType* in_ptr =
      (DType*)cl::sycl::malloc_device(32 * 1024 * 3 * sizeof(DType), q);
  auto input = sycldnn::helpers::read_binary_data(argv[2]);
  helpers::copy_to_device(input, in_ptr, q);

  Network<DType> input_transform_block;
  Network<DType> base_transform_block;
  Network<DType> feature_transform_block;

  // Construct input transformation block of network
  add_conv_bias_bnorm_relu_block(
      input_transform_block, handle, data_dir,
      "transform.input_transform.conv1.weight.bin",
      "transform.input_transform.conv1.bias.bin",
      "transform.input_transform.bn1.weight.bin",
      "transform.input_transform.bn1.bias.bin",
      "transform.input_transform.bn1.running_mean.bin",
      "transform.input_transform.bn1.running_var.bin", 32, 3, 1024, 1, 64, 1,
      1);

  add_conv_bias_bnorm_relu_block(
      input_transform_block, handle, data_dir,
      "transform.input_transform.conv2.weight.bin",
      "transform.input_transform.conv2.bias.bin",
      "transform.input_transform.bn2.weight.bin",
      "transform.input_transform.bn2.bias.bin",
      "transform.input_transform.bn2.running_mean.bin",
      "transform.input_transform.bn2.running_var.bin", 32, 64, 1024, 1, 128, 1,
      1);

  add_conv_bias_bnorm_relu_block(
      input_transform_block, handle, data_dir,
      "transform.input_transform.conv3.weight.bin",
      "transform.input_transform.conv3.bias.bin",
      "transform.input_transform.bn3.weight.bin",
      "transform.input_transform.bn3.bias.bin",
      "transform.input_transform.bn3.running_mean.bin",
      "transform.input_transform.bn3.running_var.bin", 32, 128, 1024, 1, 1024,
      1, 1);

  add_global_max_pool_layer(input_transform_block, handle, 32, 1024, 1024, 1);

  add_fc_bias_bnorm_relu_block(input_transform_block, handle, data_dir,
                               "transform.input_transform.fc1.weight.bin",
                               "transform.input_transform.fc1.bias.bin",
                               "transform.input_transform.bn4.weight.bin",
                               "transform.input_transform.bn4.bias.bin",
                               "transform.input_transform.bn4.running_mean.bin",
                               "transform.input_transform.bn4.running_var.bin",
                               32, 1024, 512);

  add_fc_bias_bnorm_relu_block(input_transform_block, handle, data_dir,
                               "transform.input_transform.fc2.weight.bin",
                               "transform.input_transform.fc2.bias.bin",
                               "transform.input_transform.bn5.weight.bin",
                               "transform.input_transform.bn5.bias.bin",
                               "transform.input_transform.bn5.running_mean.bin",
                               "transform.input_transform.bn5.running_var.bin",
                               32, 512, 256);

  add_fc_layer(input_transform_block, handle,
               data_dir + "transform.input_transform.fc3.weight.bin", 32, 256,
               9);
  add_bias_layer(input_transform_block, handle,
                 data_dir + "transform.input_transform.fc3.bias.bin", 32, 9, 1,
                 1);
  add_bias_layer(input_transform_block, handle,
                 data_dir + "transform.input_transform.id.bin", 1, 32 * 9, 1,
                 1);

  // Transform input
  add_mm_layer(input_transform_block, handle, in_ptr, 32, 1024, 3, 3);

  // Construct base transformation block
  add_conv_bias_bnorm_relu_block(
      base_transform_block, handle, data_dir, "transform.conv1.weight.bin",
      "transform.conv1.bias.bin", "transform.bn1.weight.bin",
      "transform.bn1.bias.bin", "transform.bn1.running_mean.bin",
      "transform.bn1.running_var.bin", 32, 3, 1024, 1, 64, 1, 1);

  // Construct feature transformation block
  add_conv_bias_bnorm_relu_block(
      feature_transform_block, handle, data_dir,
      "transform.feature_transform.conv1.weight.bin",
      "transform.feature_transform.conv1.bias.bin",
      "transform.feature_transform.bn1.weight.bin",
      "transform.feature_transform.bn1.bias.bin",
      "transform.feature_transform.bn1.running_mean.bin",
      "transform.feature_transform.bn1.running_var.bin", 32, 64, 1024, 1, 64, 1,
      1);
  add_conv_bias_bnorm_relu_block(
      feature_transform_block, handle, data_dir,
      "transform.feature_transform.conv2.weight.bin",
      "transform.feature_transform.conv2.bias.bin",
      "transform.feature_transform.bn2.weight.bin",
      "transform.feature_transform.bn2.bias.bin",
      "transform.feature_transform.bn2.running_mean.bin",
      "transform.feature_transform.bn2.running_var.bin", 32, 64, 1024, 1, 128,
      1, 1);
  add_conv_bias_bnorm_relu_block(
      feature_transform_block, handle, data_dir,
      "transform.feature_transform.conv3.weight.bin",
      "transform.feature_transform.conv3.bias.bin",
      "transform.feature_transform.bn3.weight.bin",
      "transform.feature_transform.bn3.bias.bin",
      "transform.feature_transform.bn3.running_mean.bin",
      "transform.feature_transform.bn3.running_var.bin", 32, 128, 1024, 1, 1024,
      1, 1);

  add_global_max_pool_layer(feature_transform_block, handle, 32, 1024, 1024, 1);

  add_fc_bias_bnorm_relu_block(
      feature_transform_block, handle, data_dir,
      "transform.feature_transform.fc1.weight.bin",
      "transform.feature_transform.fc1.bias.bin",
      "transform.feature_transform.bn4.weight.bin",
      "transform.feature_transform.bn4.bias.bin",
      "transform.feature_transform.bn4.running_mean.bin",
      "transform.feature_transform.bn4.running_var.bin", 32, 1024, 512);

  add_fc_bias_bnorm_relu_block(
      feature_transform_block, handle, data_dir,
      "transform.feature_transform.fc2.weight.bin",
      "transform.feature_transform.fc2.bias.bin",
      "transform.feature_transform.bn5.weight.bin",
      "transform.feature_transform.bn5.bias.bin",
      "transform.feature_transform.bn5.running_mean.bin",
      "transform.feature_transform.bn5.running_var.bin", 32, 512, 256);

  add_fc_layer(feature_transform_block, handle,
               data_dir + "transform.feature_transform.fc3.weight.bin", 32, 256,
               4096);
  add_bias_layer(feature_transform_block, handle,
                 data_dir + "transform.feature_transform.fc3.bias.bin", 32,
                 4096, 1, 1);
  add_bias_layer(feature_transform_block, handle,
                 data_dir + "transform.feature_transform.id.bin", 1, 32 * 4096,
                 1, 1);
  add_mm_layer(feature_transform_block, handle,
               base_transform_block.get_last_output_ptr(), 32, 1024, 64, 64);

  add_conv_bias_bnorm_relu_block(
      feature_transform_block, handle, data_dir, "transform.conv2.weight.bin",
      "transform.conv2.bias.bin", "transform.bn2.weight.bin",
      "transform.bn2.bias.bin", "transform.bn2.running_mean.bin",
      "transform.bn2.running_var.bin", 32, 64, 1024, 1, 128, 1, 1);

  add_conv_bias_bnorm_relu_block(
      feature_transform_block, handle, data_dir, "transform.conv3.weight.bin",
      "transform.conv3.bias.bin", "transform.bn3.weight.bin",
      "transform.bn3.bias.bin", "transform.bn3.running_mean.bin",
      "transform.bn3.running_var.bin", 32, 128, 1024, 1, 1024, 1, 1, false);

  add_global_max_pool_layer(feature_transform_block, handle, 32, 1024, 1, 1024);

  add_fc_bias_bnorm_relu_block(
      feature_transform_block, handle, data_dir, "fc1.weight.bin",
      "fc1.bias.bin", "bn1.weight.bin", "bn1.bias.bin", "bn1.running_mean.bin",
      "bn1.running_var.bin", 32, 1024, 512);

  add_fc_bias_bnorm_relu_block(
      feature_transform_block, handle, data_dir, "fc2.weight.bin",
      "fc2.bias.bin", "bn2.weight.bin", "bn2.bias.bin", "bn2.running_mean.bin",
      "bn2.running_var.bin", 32, 512, 256);

  add_fc_layer(feature_transform_block, handle, data_dir + "fc3.weight.bin", 32,
               256, 10);
  add_bias_layer(feature_transform_block, handle, data_dir + "fc3.bias.bin", 32,
                 10, 1, 1);
  add_softmax_layer(feature_transform_block, handle, 32, 10, 1, 1);

  add_activation_layer(feature_transform_block, handle, 32, 10, 1, 1,
                       ActivationMode::ACTIVATION_LOG);

  // Execute network in blocks
  input_transform_block.execute(in_ptr);
  base_transform_block.execute(input_transform_block.get_last_output_ptr());
  feature_transform_block.execute(base_transform_block.get_last_output_ptr());

  auto output = feature_transform_block.get_output_as_host_vec();

  // Find index of max value in each row of output, then calculate mode of
  // results to get final classification
  std::vector<int> predicted(32);
  for (int i = 0; i < 32; i++) {
    auto maxVal = std::max_element(output.begin() + (i * 10),
                                   output.begin() + (i * 10) + 10);
    predicted[i] = std::distance(output.begin() + (i * 10), maxVal);
  }
  std::sort(predicted.begin(), predicted.end());

  int prev = predicted[0];
  int count = 1;
  int mode = 0;
  int mode_count = 0;
  for (size_t i = 1; i < predicted.size(); ++i) {
    if (predicted[i] == prev) {
      count++;
    } else {
      if (count > mode_count) {
        mode = prev;
        mode_count = count;
      }
      count = 1;
    }
    prev = predicted[i];
  }
  if (count > mode_count) {
    mode = prev;
    mode_count = count;
  }
  std::cout << "classed as " << mode << " (i.e., " << object_map[mode] << ")"
            << std::endl;

  int loops = 8;
  do {
    auto start = std::chrono::high_resolution_clock::now();
    input_transform_block.execute(in_ptr);
    base_transform_block.execute(input_transform_block.get_last_output_ptr());
    feature_transform_block.execute(base_transform_block.get_last_output_ptr());
    handle.getQueue().wait_and_throw();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << (end - start).count() << " ns" << std::endl;
  } while (--loops);

  sycl::free(in_ptr, q);

  return 0;
}
