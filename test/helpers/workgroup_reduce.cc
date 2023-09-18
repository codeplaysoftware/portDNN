/*
 * Copyright Codeplay Software Ltd.
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
#include <gtest/gtest.h>

#include "portdnn/accessor_types.h"
#include "portdnn/mem_object.h"

#include "portdnn/backend/snn_backend.h"

#include "portdnn/helpers/ratio.h"
#include "portdnn/helpers/scope_exit.h"

#include "src/helpers/flattened_id.h"
#include "src/helpers/vector_io.h"
#include "src/helpers/vector_type.h"
#include "src/helpers/workgroup_reduce.h"

#include "test/backend/backend_test_fixture.h"

#include "test/gen/iota_initialised_data.h"

#include "test/helpers/float_comparison.h"

#include "test/types/kernel_data_types.h"
#include "test/types/to_gtest_types.h"

#include <stddef.h>
#include <string>
#include <utility>
#include <vector>

#include <CL/sycl.hpp>

namespace {

/**
 * Kernel to load data and reduce across workgroup.
 *
 * Each work item loads a vector of given Width, then uses that to reduce across
 * the workgroup. The first item in each workgroup then writes the result to the
 * output buffer. The vector is not reduced to a scalar, rather the reduction is
 * performed while preserving the vector's shape.
 *
 * \tparam T     Scalar data type to load and reduce
 * \tparam Width Width of vector to use in kernel
 * \tparam Dims  Dimensions of workgroup
 */
template <typename T, int Width, int Dims>
struct Reducer {
  using VecType = typename sycldnn::helpers::VectorType<T, Width>::type;
  using Load = sycldnn::helpers::io::Load<VecType>;
  using Store = sycldnn::helpers::io::Store<VecType>;

  void operator()(cl::sycl::nd_item<Dims> item) const {
    size_t lin_idx = sycldnn::helpers::get_flattened_global_id(item) * Width;
    if (lin_idx < data_size) {
      auto input_ptr =
          sycldnn::helpers::internal::as_const_ptr(input.get_pointer());
      auto data = Load()(input_ptr, lin_idx);
      data = sycldnn::helpers::reduce::workgroup_reduce<
          sycldnn::helpers::reduce::Sum, size_t>(
          data, item,
          workspace.template get_multi_ptr<sycl::access::decorated::legacy>());
      if (sycldnn::helpers::get_flattened_local_id(item) == 0) {
        size_t group_id = sycldnn::helpers::get_flattened_group_id(item);
        Store()(output.get_pointer(), group_id * Width, data);
      }
    }
  }
  sycldnn::ReadAccessor<T> input;
  sycldnn::WriteAccessor<T> output;
  sycldnn::LocalAccessor<T> workspace;
  size_t data_size;
};

}  // namespace

/**
 * Workgroup reduction test fixture.
 *
 * Provides a `test` method that will allocate memory, run a reduction kernel
 * and check the results against the expected results. This uses an SNNBackend
 * and associated BackendProvider to generate the buffers and SYCL objects
 * required.
 *
 * \tparam T     Scalar data type to load and reduce
 * \tparam Width Width of vector to use in kernel
 */
template <typename T, int Width>
struct WorkspaceReductionTest
    : public BackendTestFixture<sycldnn::backend::SNNBackend> {
  using ScalarType = T;

  template <int Dims>
  void test(cl::sycl::range<Dims> data_sizes,
            cl::sycl::range<Dims> workgroup_sizes,
            std::vector<ScalarType> const& exp) {
    auto& provider = this->provider_;
    auto& backend = provider.get_backend();
    auto device = backend.get_queue().get_device();

    size_t max_workgroup_dims = device.template get_info<
        cl::sycl::info::device::max_work_item_dimensions>();
    if (Dims > max_workgroup_dims) {
      GTEST_SKIP() << "Skipping test because the hardware does not support a "
                      "workgroup with this many dimensions.";
    }
    auto max_workitem_sizes =
#ifndef SYCL_IMPLEMENTATION_ONEAPI
        device.template get_info<cl::sycl::info::device::max_work_item_sizes>();
#else
        device.template get_info<
            cl::sycl::info::device::max_work_item_sizes<3> >();
#endif

    for (int i = 0; i < Dims; ++i) {
      if (workgroup_sizes[i] > max_workitem_sizes[i]) {
        GTEST_SKIP() << "Skipping test because the hardware does not support "
                        "this many items in the "
                     << i << " dimensions.";
      }
    }
    size_t total_workgroup_size = workgroup_sizes.size();
    size_t max_workgroup_size =
        device.template get_info<cl::sycl::info::device::max_work_group_size>();
    if (total_workgroup_size > max_workgroup_size) {
      GTEST_SKIP() << "Skipping test because the hardware does not support "
                      "this workgroup size.";
    }

    size_t flat_size = data_sizes.size();
    size_t n_workgroups = sycldnn::helpers::round_ratio_up_above_zero(
        flat_size, total_workgroup_size);
    size_t out_size = n_workgroups * Width;
    size_t in_size = flat_size * Width;

    ASSERT_EQ(exp.size(), out_size);

    auto input =
        iota_initialised_data(in_size, static_cast<ScalarType const>(in_size));
    auto output = std::vector<ScalarType>(out_size);

    auto inp_gpu = provider.get_initialised_device_memory(in_size, input);
    auto out_gpu = provider.get_initialised_device_memory(out_size, output);
    SNN_ON_SCOPE_EXIT {
      provider.deallocate_ptr(inp_gpu);
      provider.deallocate_ptr(out_gpu);
    };
    auto in_mem = backend.get_mem_object(inp_gpu, in_size);
    auto out_mem = backend.get_mem_object(out_gpu, out_size);

    auto event = backend.get_queue().submit([&](cl::sycl::handler& cgh) {
      auto in_acc = in_mem.read_accessor(cgh);
      auto out_acc = out_mem.write_accessor(cgh);
      sycldnn::LocalAccessor<ScalarType> local_acc{
          cl::sycl::range<1>{total_workgroup_size * Width}, cgh};

      auto functor =
          Reducer<ScalarType, Width, Dims>{in_acc, out_acc, local_acc, in_size};
      cgh.parallel_for(cl::sycl::nd_range<Dims>{data_sizes, workgroup_sizes},
                       functor);
    });
    event.wait_and_throw();
    provider.copy_device_data_to_host(out_size, out_gpu, output);
    for (size_t i = 0; i < exp.size(); ++i) {
      SCOPED_TRACE("Element: " + std::to_string(i));
      SNN_ALMOST_EQUAL(exp[i], output[i], 64u);
    }
  }
};

using DataTypes =
    sycldnn::types::ToGTestTypes<sycldnn::types::KernelDataTypes>::type;

template <typename T>
using WorkspaceReductionTestScalar = WorkspaceReductionTest<T, 1>;
TYPED_TEST_SUITE(WorkspaceReductionTestScalar, DataTypes);

TYPED_TEST(WorkspaceReductionTestScalar, SingleOut16) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<1>{16};
  auto workgroup = cl::sycl::range<1>{16};
  // When the workgroup size matches the data size, the full input array is
  // reduced to a single value. Assuming the data is initialised using `iota`,
  // this value is therefore the sum from 1 to the number of elements in the
  // input; in this case the sum 1 to 16.
  auto exp = std::vector<DataType>{136};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestScalar, SingleOut8x2) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<2>{8, 2};
  auto workgroup = cl::sycl::range<2>{8, 2};
  auto exp = std::vector<DataType>{136};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestScalar, SingleOut4x2x2) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<3>{4, 2, 2};
  auto workgroup = cl::sycl::range<3>{4, 2, 2};
  auto exp = std::vector<DataType>{136};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestScalar, SingleOut128) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<1>{128};
  auto workgroup = cl::sycl::range<1>{128};
  auto exp = std::vector<DataType>{8256};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestScalar, SingleOut16x8) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<2>{16, 8};
  auto workgroup = cl::sycl::range<2>{16, 8};
  auto exp = std::vector<DataType>{8256};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestScalar, SingleOut4x4x8) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<3>{4, 4, 8};
  auto workgroup = cl::sycl::range<3>{4, 4, 8};
  auto exp = std::vector<DataType>{8256};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestScalar, FourOut16) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<1>{16 * 4};
  auto workgroup = cl::sycl::range<1>{16};
  // Each workgroup sums up 16 contiguous numbers.
  // Workgroup 1's output = sum 1 to 16
  // Workgroup 2's output = sum 17 to 32
  auto exp = std::vector<DataType>{136, 392, 648, 904};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestScalar, FourOut8x2Last) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<2>{8, 2 * 4};
  auto workgroup = cl::sycl::range<2>{8, 2};
  // Here each workgroup operates on a contiguous block of memory, as the data
  // size is only multiplied in the last dimension, so the result is the same as
  // for the 1D workgroup in `FourOut16`.
  auto exp = std::vector<DataType>{136, 392, 648, 904};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestScalar, FourOut8x2) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<2>{8 * 2, 2 * 2};
  auto workgroup = cl::sycl::range<2>{8, 2};
  // Each workgroup operates over slices of the data:
  //
  //   <---- workgroup 1 ----> <---- workgroup 2 ---->
  //    1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
  //   17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
  //
  //   <---- workgroup 3 ----> <---- workgroup 4 ---->
  //   33 34 35 36 ...
  //   49 50 51 52 ...
  //
  // Workgroup 1's output = (sum 1 to 8) + (sum 17 to 24)
  // Workgroup 2's output = (sum 9 to 16) + (sum 25 to 32)
  // Workgroup 3's output = (sum 33 to 40) + (sum 49 to 56)
  // Workgroup 4's output = (sum 41 to 48) + (sum 57 to 64)
  auto exp = std::vector<DataType>{200, 328, 712, 840};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestScalar, FourOut4x2x2Last) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<3>{4, 2, 2 * 4};
  auto workgroup = cl::sycl::range<3>{4, 2, 2};
  // Here each workgroup operates on a contiguous block of memory, as the data
  // size is only multiplied in the last dimension, so the result is the same as
  // for the 1D workgroup in `FourOut16`.
  auto exp = std::vector<DataType>{136, 392, 648, 904};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestScalar, FourOut4x2x2) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<3>{4, 2 * 2, 2 * 2};
  auto workgroup = cl::sycl::range<3>{4, 2, 2};
  // As the data is multiplied in the middle dimension each of the four
  // workgroups has to compute two slices over the data, each of which is (4x2)
  // elements wide.
  // This gives the same output as `FourOut8x2`.
  auto exp = std::vector<DataType>{200, 328, 712, 840};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestScalar, FourOut4x2x2Alt) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<3>{4 * 2, 2 * 2, 2};
  auto workgroup = cl::sycl::range<3>{4, 2, 2};
  // Each workgroup computes only 4 elements contiguously, as there is more data
  // in the first dimension than there are items in the workgroup. As a result,
  // each output is the sum of the reduction of a number of four element blocks.
  //
  // WG1: (sum  1 to  4) + (sum  9 to 12) + (sum 33 to 36) + (sum 41 to 44)
  // WG2: (sum  5 to  8) + (sum 13 to 16) + (sum 37 to 40) + (sum 45 to 48)
  // WG3: (sum 17 to 20) + (sum 25 to 28) + (sum 49 to 52) + (sum 57 to 60)
  // WG4: (sum 21 to 24) + (sum 29 to 32) + (sum 53 to 56) + (sum 61 to 64)
  auto exp = std::vector<DataType>{360, 424, 616, 680};
  this->test(size, workgroup, exp);
}

template <typename T>
using WorkspaceReductionTestVec2 = WorkspaceReductionTest<T, 2>;
TYPED_TEST_SUITE(WorkspaceReductionTestVec2, DataTypes);

TYPED_TEST(WorkspaceReductionTestVec2, SingleOut16) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<1>{16};
  auto workgroup = cl::sycl::range<1>{16};
  // With vectors the actual input data size must be multiplied by the number of
  // elements in the vector, as each workitem will read that many scalars at
  // once. Here we use an input with values from 1 to 32 and the output elements
  // will reduce every other value to give the output. The first element takes
  // values 1, 3, 5, ..., 31 and the second takes 2, 4, ..., 32.
  auto exp = std::vector<DataType>{256, 272};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestVec2, SingleOut8x2) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<2>{8, 2};
  auto workgroup = cl::sycl::range<2>{8, 2};
  auto exp = std::vector<DataType>{256, 272};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestVec2, SingleOut4x2x2) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<3>{4, 2, 2};
  auto workgroup = cl::sycl::range<3>{4, 2, 2};
  auto exp = std::vector<DataType>{256, 272};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestVec2, SingleOut128) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<1>{128};
  auto workgroup = cl::sycl::range<1>{128};
  auto exp = std::vector<DataType>{16384, 16512};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestVec2, SingleOut16x8) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<2>{16, 8};
  auto workgroup = cl::sycl::range<2>{16, 8};
  auto exp = std::vector<DataType>{16384, 16512};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestVec2, SingleOut4x4x8) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<3>{4, 4, 8};
  auto workgroup = cl::sycl::range<3>{4, 4, 8};
  auto exp = std::vector<DataType>{16384, 16512};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestVec2, FourOut16) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<1>{16 * 4};
  auto workgroup = cl::sycl::range<1>{16};
  // Each workgroup sums up 16 vectors, made up of continguous numbers.
  // Workgroup 1's output 1 = sum 1 to 32 step 2
  // Workgroup 1's output 2 = sum 2 to 32 step 2
  // Workgroup 2's output 1 = sum 33 to 64 step 2
  // Workgroup 2's output 2 = sum 34 to 64 step 2
  auto exp = std::vector<DataType>{256, 272, 768, 784, 1280, 1296, 1792, 1808};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestVec2, FourOut8x2Last) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<2>{8, 2 * 4};
  auto workgroup = cl::sycl::range<2>{8, 2};
  // Here each workgroup operates on a contiguous block of memory, as the data
  // size is only multiplied in the last dimension, so the result is the same as
  // the 1D workgroup.
  auto exp = std::vector<DataType>{256, 272, 768, 784, 1280, 1296, 1792, 1808};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestVec2, FourOut8x2) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<2>{8 * 2, 2 * 2};
  auto workgroup = cl::sycl::range<2>{8, 2};
  // Each workgroup operates over 16 element slices of the data:
  //
  // Workgroup 1's output 1 = (sum  1 to 16 st 2) + (sum 33 to 48 st 2)
  // Workgroup 1's output 2 = (sum  2 to 16 st 2) + (sum 34 to 48 st 2)
  // Workgroup 2's output 1 = (sum 17 to 32 st 2) + (sum 49 to 64 st 2)
  // Workgroup 2's output 2 = (sum 18 to 32 st 2) + (sum 50 to 64 st 2)
  // Workgroup 3's output 1 = (sum 65 to 80 st 2) + (sum 97 to 112 st 2)
  // Workgroup 3's output 2 = (sum 66 to 80 st 2) + (sum 98 to 112 st 2)
  // Workgroup 4's output 1 = (sum 81 to 96 st 2) + (sum 113 to 128 st 2)
  // Workgroup 4's output 2 = (sum 82 to 96 st 2) + (sum 114 to 128 st 2)
  auto exp = std::vector<DataType>{384, 400, 640, 656, 1408, 1424, 1664, 1680};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestVec2, FourOut4x2x2Last) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<3>{4, 2, 2 * 4};
  auto workgroup = cl::sycl::range<3>{4, 2, 2};
  // Here each workgroup operates on a contiguous block of memory, as the data
  // size is only multiplied in the last dimension, so the result is the same as
  // the 1D workgroup.
  auto exp = std::vector<DataType>{256, 272, 768, 784, 1280, 1296, 1792, 1808};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestVec2, FourOut4x2x2) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<3>{4, 2 * 2, 2 * 2};
  auto workgroup = cl::sycl::range<3>{4, 2, 2};
  // As the data is multiplied in the middle dimension each of the four
  // workgroups has to compute two slices over the data, each of which is (4x2)
  // elements wide.
  // This gives the same output as FourOut8x2.
  auto exp = std::vector<DataType>{384, 400, 640, 656, 1408, 1424, 1664, 1680};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestVec2, FourOut4x2x2Alt) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<3>{4 * 2, 2 * 2, 2};
  auto workgroup = cl::sycl::range<3>{4, 2, 2};
  // Each workgroup computes only 4 elements contiguously, as there is more data
  // in the first dimension than there are items in the workgroup. As a result,
  // each output is the sum of the reduction of a number of four element blocks.
  //
  // The first elements of the vectors are given by the following, where the
  // sums include a step of 2. The second elements are given by the sum of the
  // elements that were stepped over.
  // WG1: (sum  1 to  8) + (sum 17 to 24) + (sum  65 to  72) + (sum  81 to  88)
  // WG2: (sum  9 to 16) + (sum 25 to 32) + (sum  73 to  80) + (sum  89 to  96)
  // WG3: (sum 33 to 40) + (sum 49 to 56) + (sum  97 to 104) + (sum 113 to 120)
  // WG4: (sum 41 to 48) + (sum 57 to 64) + (sum 105 to 112) + (sum 121 to 128)
  auto exp = std::vector<DataType>{704, 720, 832, 848, 1216, 1232, 1344, 1360};
  this->test(size, workgroup, exp);
}

template <typename T>
using WorkspaceReductionTestVec4 = WorkspaceReductionTest<T, 4>;
TYPED_TEST_SUITE(WorkspaceReductionTestVec4, DataTypes);

TYPED_TEST(WorkspaceReductionTestVec4, SingleOut16) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<1>{16};
  auto workgroup = cl::sycl::range<1>{16};
  auto exp = std::vector<DataType>{496, 512, 528, 544};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestVec4, SingleOut8x2) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<2>{8, 2};
  auto workgroup = cl::sycl::range<2>{8, 2};
  auto exp = std::vector<DataType>{496, 512, 528, 544};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestVec4, SingleOut4x2x2) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<3>{4, 2, 2};
  auto workgroup = cl::sycl::range<3>{4, 2, 2};
  auto exp = std::vector<DataType>{496, 512, 528, 544};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestVec4, SingleOut128) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<1>{128};
  auto workgroup = cl::sycl::range<1>{128};
  auto exp = std::vector<DataType>{32640, 32768, 32896, 33024};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestVec4, SingleOut16x8) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<2>{16, 8};
  auto workgroup = cl::sycl::range<2>{16, 8};
  auto exp = std::vector<DataType>{32640, 32768, 32896, 33024};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestVec4, SingleOut4x4x8) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<3>{4, 4, 8};
  auto workgroup = cl::sycl::range<3>{4, 4, 8};
  auto exp = std::vector<DataType>{32640, 32768, 32896, 33024};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestVec4, FourOut16) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<1>{16 * 4};
  auto workgroup = cl::sycl::range<1>{16};
  // Each workgroup sums up 16 vectors, made up of continguous numbers.
  // Workgroup 1's output 1 = sum 1 to 64 step 4
  // Workgroup 1's output 2 = sum 2 to 64 step 4
  // Workgroup 1's output 3 = sum 3 to 64 step 4
  // Workgroup 1's output 4 = sum 4 to 64 step 4
  // Workgroup 2's output 1 = sum 65 to 128 step 4
  // Workgroup 2's output 2 = sum 66 to 128 step 4
  auto exp =
      std::vector<DataType>{496,  512,  528,  544,  1520, 1536, 1552, 1568,
                            2544, 2560, 2576, 2592, 3568, 3584, 3600, 3616};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestVec4, FourOut8x2Last) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<2>{8, 2 * 4};
  auto workgroup = cl::sycl::range<2>{8, 2};
  // Here each workgroup operates on a contiguous block of memory, as the data
  // size is only multiplied in the last dimension, so the result is the same as
  // the 1D workgroup.
  auto exp =
      std::vector<DataType>{496,  512,  528,  544,  1520, 1536, 1552, 1568,
                            2544, 2560, 2576, 2592, 3568, 3584, 3600, 3616};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestVec4, FourOut8x2) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<2>{8 * 2, 2 * 2};
  auto workgroup = cl::sycl::range<2>{8, 2};
  // Each workgroup operates over 16 element slices of the data:
  //
  // Workgroup 1's output 1 = (sum   1 to  32 st 4) + (sum  65 to  96 st 4)
  // Workgroup 2's output 1 = (sum  33 to  64 st 4) + (sum  97 to 128 st 4)
  // Workgroup 3's output 1 = (sum 129 to 160 st 4) + (sum 193 to 224 st 4)
  // Workgroup 4's output 1 = (sum 161 to 192 st 4) + (sum 225 to 256 st 4)
  auto exp =
      std::vector<DataType>{752,  768,  784,  800,  1264, 1280, 1296, 1312,
                            2800, 2816, 2832, 2848, 3312, 3328, 3344, 3360};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestVec4, FourOut4x2x2Last) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<3>{4, 2, 2 * 4};
  auto workgroup = cl::sycl::range<3>{4, 2, 2};
  // Here each workgroup operates on a contiguous block of memory, as the data
  // size is only multiplied in the last dimension, so the result is the same as
  // the 1D workgroup.
  auto exp =
      std::vector<DataType>{496,  512,  528,  544,  1520, 1536, 1552, 1568,
                            2544, 2560, 2576, 2592, 3568, 3584, 3600, 3616};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestVec4, FourOut4x2x2) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<3>{4, 2 * 2, 2 * 2};
  auto workgroup = cl::sycl::range<3>{4, 2, 2};
  // As the data is multiplied in the middle dimension each of the four
  // workgroups has to compute two slices over the data, each of which is (4x2)
  // elements wide.
  // This gives the same output as FourOut8x2.
  auto exp =
      std::vector<DataType>{752,  768,  784,  800,  1264, 1280, 1296, 1312,
                            2800, 2816, 2832, 2848, 3312, 3328, 3344, 3360};
  this->test(size, workgroup, exp);
}

TYPED_TEST(WorkspaceReductionTestVec4, FourOut4x2x2Alt) {
  using DataType = typename TestFixture::ScalarType;
  auto size = cl::sycl::range<3>{4 * 2, 2 * 2, 2};
  auto workgroup = cl::sycl::range<3>{4, 2, 2};
  // Each workgroup computes only 4 elements contiguously, as there is more data
  // in the first dimension than there are items in the workgroup. As a result,
  // each output is the sum of the reduction of a number of four element blocks.
  //
  //  <----- wg 1 ---->  <----- wg 2 ---->
  //   1   5   9  13     17  21  25  29
  //    2   6  10  14     18  22  26  30
  //     3   7  11  15     19  23  27  31
  //      4   8  12  16     20  24  28  32
  //
  //   33  37  41  45     49  53  57  61
  //    34   ...    46     50   ...    62
  //     35   ...    47     51   ...    63
  //      36   ...    48     52   ...    64
  //
  // The first elements of the vectors are given by the following, where the
  // sums include a step of 4. The second, third and fourth elements are given
  // by the sums of the next elements that were stepped over. As each sum is
  // over 16 numbers the difference between the first and second output is 16.
  //
  // WG1 (sum  1 to 16) + (sum  33 to  48) + (sum 129 to 144) + (sum 161 to 176)
  // WG2 (sum 17 to 32) + (sum  49 to  64) + (sum 145 to 160) + (sum 177 to 192)
  // WG3 (sum 65 to 80) + (sum  97 to 112) + (sum 193 to 208) + (sum 225 to 240)
  // WG4 (sum 81 to 96) + (sum 113 to 128) + (sum 209 to 224) + (sum 241 to 256)
  auto exp =
      std::vector<DataType>{1392, 1408, 1424, 1440, 1648, 1664, 1680, 1696,
                            2416, 2432, 2448, 2464, 2672, 2688, 2704, 2720};
  this->test(size, workgroup, exp);
}
