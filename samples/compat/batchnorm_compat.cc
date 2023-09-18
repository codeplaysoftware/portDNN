// original sample from
// https://github.com/kaixih/dl_samples/blob/main/batch_norm/cudnn_batch_norm.cu
#include <iostream>
#include <portdnn/compat/batchnorm.hpp>

using namespace sycldnn::compat;

void print_array(float* array, int size, const char* name) {
  std::cout << name;
  for (int i = 0; i < size; i++) {
    std::cout << array[i] << " ";
  }
  std::cout << std::endl;
}

int main() {
  SNNHandle handle;
  SNNCreate(handle);
  sycl::queue q = handle.getQueue();

  auto mode = BatchNormMode::BATCHNORM_SPATIAL;

  float one = 1.0;
  float zero = 0.0;
  int N = 2, C = 3, H = 1, W = 2;

  int x_size = N * C * H * W;
  int x_size_bytes = x_size * sizeof(float);

  int mean_size = C;
  int mean_size_bytes = mean_size * sizeof(float);

  TensorDescriptor x_descriptor;

  x_descriptor.set4d(
      /*format=*/sycldnn::DataFormat::NCHW,
      /*batch_size=*/N,
      /*channels=*/C,
      /*image_height=*/H,
      /*image_width=*/W);

  float* x = (float*)sycl::malloc_shared(x_size_bytes, q);
  float* y = (float*)sycl::malloc_shared(x_size_bytes, q);
  float* dy = (float*)sycl::malloc_shared(x_size_bytes, q);
  float* dx = (float*)sycl::malloc_shared(x_size_bytes, q);

  x[0] = 0.16513085;
  x[2] = 0.9014813;
  x[4] = 0.6309742;
  x[1] = 0.4345461;
  x[3] = 0.29193902;
  x[5] = 0.64250207;
  x[6] = 0.9757855;
  x[8] = 0.43509948;
  x[10] = 0.6601019;
  x[7] = 0.60489583;
  x[9] = 0.6366315;
  x[11] = 0.6144488;

  dy[0] = 1.0;
  dy[2] = 1.0;
  dy[4] = 1.0;
  dy[1] = 1.0;
  dy[3] = 1.0;
  dy[5] = 1.0;
  dy[6] = 1.0;
  dy[8] = 1.0;
  dy[10] = 1.0;
  dy[7] = 1.0;
  dy[9] = 1.0;
  dy[11] = 1.0;

  TensorDescriptor mean_descriptor;

  mean_descriptor.set4d(
      /*format=*/sycldnn::DataFormat::NCHW,
      /*batch_size=*/1,
      /*channels=*/C,
      /*image_height=*/1,
      /*image_width=*/1);

  float* scale = (float*)sycl::malloc_shared(mean_size_bytes, q);
  float* offset = (float*)sycl::malloc_shared(mean_size_bytes, q);
  float* dscale = (float*)sycl::malloc_shared(mean_size_bytes, q);
  float* doffset = (float*)sycl::malloc_shared(mean_size_bytes, q);
  float* running_mean = (float*)sycl::malloc_shared(mean_size_bytes, q);
  float* running_var = (float*)sycl::malloc_shared(mean_size_bytes, q);
  float* saved_mean = (float*)sycl::malloc_shared(mean_size_bytes, q);
  float* saved_inv_var = (float*)sycl::malloc_shared(mean_size_bytes, q);

  scale[0] = 1.0;
  scale[1] = 1.0;
  scale[2] = 1.0;
  offset[0] = 0.0;
  offset[1] = 0.0;
  offset[2] = 0.0;

  running_mean[0] = 1.0;
  running_mean[1] = 1.0;
  running_mean[2] = 1.0;
  running_var[0] = 1.0;
  running_var[1] = 1.0;
  running_var[2] = 1.0;

  batchNormalizationForwardTraining(
      /*handle=*/handle,
      /*mode=*/mode,
      /*alpha=*/&one,
      /*beta=*/&zero,
      /*xDesc=*/x_descriptor,
      /*xData=*/x,
      /*yDesc=*/x_descriptor,
      /*yData=*/y,
      /*bnScaleBiasMeanVarDesc=*/mean_descriptor,
      /*bnScale=*/scale,
      /*bnBias=*/offset,
      /*exponentialAverageFactor=*/0.5,
      /*resultRunningMean=*/running_mean,
      /*resultRunningVariance=*/running_var,
      /*epsilon=*/0.001,
      /*resultSaveMean=*/saved_mean,
      /*resultSaveInvVariance=*/saved_inv_var);

  q.wait();

  print_array(y, x_size, "y NCHW format: ");
  print_array(saved_mean, mean_size, "saved MEAN: ");
  print_array(saved_inv_var, mean_size, "saved inv var: ");

  batchNormalizationBackward(
      /*handle=*/handle,
      /*mode=*/mode,
      /*alphaDataDiff=*/&one,
      /*betaDataDiff=*/&zero,
      /*alphaParamDiff=*/&one,
      /*betaParamDiff=*/&zero,
      /*xDesc=*/x_descriptor,
      /*xData=*/x,
      /*dyDesc=*/x_descriptor,
      /*dyData=*/dy,
      /*dxDesc=*/x_descriptor,
      /*dxData=*/dx,
      /*dBnScaleBiasDesc=*/mean_descriptor,
      /*bnScaleData=*/scale,
      /*dBnScaleData=*/dscale,
      /*dBnBiasData=*/doffset,
      /*epsilon=*/0.001,
      /*savedMean=*/saved_mean,
      /*savedInvVariance=*/saved_inv_var);

  q.wait();

  print_array(dx, x_size, "dx NCHW format: ");
  print_array(dscale, mean_size, "dscale: ");
  print_array(doffset, mean_size, "doffset: ");

  sycl::free(x, q);
  sycl::free(y, q);
  sycl::free(dy, q);
  sycl::free(dx, q);
  sycl::free(scale, q);
  sycl::free(offset, q);
  sycl::free(dscale, q);
  sycl::free(doffset, q);
  sycl::free(running_mean, q);
  sycl::free(running_var, q);
  sycl::free(saved_mean, q);
  sycl::free(saved_inv_var, q);
}
