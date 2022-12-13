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
#include "test/types/kernel_data_types.h"

#include "test/roi_align/roi_align_event_dependencies_fixture.h"

using GTestTypeList = sycldnn::types::GTestKernelDataTypes;
template <typename params>
using RoiAlignTestFixtureEventDeps = RoiAlignFixtureEventDependencies<params>;

TYPED_TEST_SUITE(RoiAlignTestFixtureEventDeps, GTestTypeList);
TYPED_TEST(RoiAlignTestFixtureEventDeps, EVENT_DEPENDENCIES) {
  sycldnn::roi_align::RoiAlignParams params;
  params.batch = 1;
  params.channels = 1;
  params.in_height = 2;
  params.in_width = 2;
  params.out_height = 1;
  params.out_width = 1;
  params.num_rois = 1;
  params.sampling_ratio = 0;
  params.spatial_scale = 1.0f;
  this->test_roi_align_event_dependencies(params);
}
