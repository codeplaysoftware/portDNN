/*
 * Copyright Codeplay Software Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use these files except in compliance with the License.
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

// Original cuDNN sample from
// https://gist.github.com/odashi/1c20ba90388cf02330e1b95963d78039

#include <CL/sycl.hpp>

#include "portdnn/backend/snn_backend.h"
#include "portdnn/compat/convolution.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace sycldnn::compat;

void dev_const(float* px, float k, const sycl::nd_item<1>& item) {
  int tid = item.get_global_linear_id();
  px[tid] = k;
}

void dev_iota(float* px, const sycl::nd_item<1>& item) {
  int tid = item.get_global_linear_id();
  px[tid] = tid;
}

void print(const float* data, int n, int c, int h, int w, sycl::queue q) {
  std::vector<float> buffer(1 << 20);
  q.memcpy(buffer.data(), data, n * c * h * w * sizeof(float)).wait();
  int a = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      std::cout << "n=" << i << ", c=" << j << ":" << std::endl;
      for (int k = 0; k < h; ++k) {
        for (int l = 0; l < w; ++l) {
          std::cout << std::setw(8) << std::right << buffer[a];
          ++a;
        }
        std::cout << std::endl;
      }
    }
  }
  std::cout << std::endl;
}

int main() {
  SNNHandle handle;
  SNNCreate(handle);

  // input
  const int in_n = 1;
  const int in_c = 1;
  const int in_h = 5;
  const int in_w = 5;
  std::cout << "in_n: " << in_n << std::endl;
  std::cout << "in_c: " << in_c << std::endl;
  std::cout << "in_h: " << in_h << std::endl;
  std::cout << "in_w: " << in_w << std::endl;
  std::cout << std::endl;

  TensorDescriptor in_desc;
  in_desc.set4d(sycldnn::DataFormat::NCHW, in_n, in_c, in_h, in_w);

  auto q = handle.getQueue();
  float* in_data =
      (float*)sycl::malloc_device(in_n * in_c * in_h * in_w * sizeof(float), q);

  // filter
  const int filt_k = 1;
  const int filt_c = 1;
  const int filt_h = 2;
  const int filt_w = 2;
  std::cout << "filt_k: " << filt_k << std::endl;
  std::cout << "filt_c: " << filt_c << std::endl;
  std::cout << "filt_h: " << filt_h << std::endl;
  std::cout << "filt_w: " << filt_w << std::endl;
  std::cout << std::endl;

  FilterDescriptor filt_desc;

  filt_desc.set4d(sycldnn::DataFormat::NCHW, filt_k, filt_c, filt_h, filt_w);

  float* filt_data = (float*)sycl::malloc_device(
      filt_k * filt_c * filt_h * filt_w * sizeof(float), q);

  // convolution
  const int pad_h = 1;
  const int pad_w = 1;
  const int str_h = 1;
  const int str_w = 1;
  const int dil_h = 1;
  const int dil_w = 1;
  std::cout << "pad_h: " << pad_h << std::endl;
  std::cout << "pad_w: " << pad_w << std::endl;
  std::cout << "str_h: " << str_h << std::endl;
  std::cout << "str_w: " << str_w << std::endl;
  std::cout << "dil_h: " << dil_h << std::endl;
  std::cout << "dil_w: " << dil_w << std::endl;
  std::cout << std::endl;

  ConvolutionDescriptor conv_desc;
  setConvolution2dDescriptor(conv_desc, pad_h, pad_w, str_h, str_w, dil_h,
                             dil_w, ConvolutionMode::CROSS_CORRELATION);

  // output
  int out_n;
  int out_c;
  int out_h;
  int out_w;

  getConvolution2dForwardOutputDim(conv_desc, in_desc, filt_desc, &out_n,
                                   &out_c, &out_h, &out_w);

  std::cout << "out_n: " << out_n << std::endl;
  std::cout << "out_c: " << out_c << std::endl;
  std::cout << "out_h: " << out_h << std::endl;
  std::cout << "out_w: " << out_w << std::endl;
  std::cout << std::endl;

  TensorDescriptor out_desc;
  out_desc.set4d(sycldnn::DataFormat::NCHW, out_n, out_c, out_h, out_w);

  float* out_data = (float*)sycl::malloc_device(
      out_n * out_c * out_h * out_w * sizeof(float), q);

  float alpha = 0.95f;
  float beta = 0.05f;

  q.parallel_for(sycl::nd_range<1>(in_w * in_h, in_n * in_c),
                 [=](sycl::nd_item<1> item) { dev_iota(in_data, item); });

  q.parallel_for(
      sycl::nd_range<1>(filt_w * filt_h, filt_k * filt_c),
      [=](sycl::nd_item<1> item) { dev_const(filt_data, 1.f, item); });

  q.parallel_for(sycl::nd_range<1>(out_w * out_h, out_n * out_c),
                 [=](sycl::nd_item<1> item) { dev_iota(out_data, item); });

  auto status =
      convolutionForward(handle, &alpha, in_desc, in_data, filt_desc, filt_data,
                         conv_desc, sycldnn::conv2d::Algorithm::Direct, nullptr,
                         0, &beta, out_desc, out_data);

  if (status.status != sycldnn::StatusCode::OK) {
    std::cout << "Error occurred when during convolution operation.\n";
    std::exit(1);
  }

  status.event.wait_and_throw();

  // results
  std::cout << "in_data:" << std::endl;
  print(in_data, in_n, in_c, in_h, in_w, q);

  std::cout << "filt_data:" << std::endl;
  print(filt_data, filt_k, filt_c, filt_h, filt_w, q);

  std::cout << "out_data:" << std::endl;
  print(out_data, out_n, out_c, out_h, out_w, q);

  return 0;
}
