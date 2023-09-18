#!/usr/bin/env python
#
# Copyright Codeplay Software Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use these files except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# Automatically generate the ScatterND test cases using TensorFlow to provide
# the expected values.

from __future__ import print_function

import itertools
import os
from collections import namedtuple

import tensorflow as tf
from tensorflow.python.framework.ops import get_gradient_function
import numpy as np

import helpers

BATCHES = [1, 3]
CHANNELS = [1, 5, 8]
ROWS = [1, 2, 8]
COLUMNS = [1, 2, 8]
TEST_TYPES = ["scatter_nd_assign", "scatter_nd_add"]
INDEX_DEPTHS = [1, 2, 3, 4]
INDEX_DEPTH_MAP = [
    "tensor_slice",
    "matrix_slice",
    "vector_slice",
    "elementwise"]
INCLUDES = r"""
#include <gtest/gtest.h>

#include "portdnn/scatter_nd/operators.h"
#include "portdnn/scatter_nd/params.h"

#include "test/scatter_nd/scatter_nd_fixture.h"
#include "test/types/cartesian_product.h"
#include "test/types/kernel_data_types.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"

#include <vector>"""
TYPED_TEST_CASE_DECL_TPL = r"""
using DataTypeList = sycldnn::types::KernelDataTypes;
using Backends = sycldnn::types::DefaultBackendTypes;

using TypeBackendPairs =
    sycldnn::types::CartesianProduct<DataTypeList, Backends>::type;

using GTestTypePairs = sycldnn::types::ToGTestTypes<TypeBackendPairs>::type;

using namespace sycldnn; // NOLINT(google-build-using-namespace)
template <typename Pair>
using {test_case} = ScatterNDFixture<Pair, int, {operator}>;
TYPED_TEST_CASE({test_case}, GTestTypePairs);"""

TestCaseParams = namedtuple('TestCaseParams', ['test_type'])
TestParams = namedtuple('TestParams', ['in_shape', 'index_depth'])

TENSORFLOW_OPS_MAP = {
    'scatter_nd_assign': tf.tensor_scatter_nd_update,
    'scatter_nd_add': tf.tensor_scatter_nd_add,
}


def get_results(max_val, index_depth, scatter_nd_op, input_shape):
    """
    Construct and run a Tensorflow graph to compute ScatterND op.

    Will create an input tensor of the required size filled with values 1, 2,
    3... and use these to compute the ScatterND op. Returns the computed values
    in a numpy array.
    """
    rng = np.random.default_rng(12345)
    # Total size of input/output tensors
    total_size = np.product(input_shape)
    # Size of each slice
    slice_size = int(np.product(input_shape[index_depth:]))
    # Total number of potential slices in the tensor
    num_slices = int(total_size / slice_size)

    # Randomly choose to fill anywhere up to just above half of the tensor
    num_updates = rng.choice(int(0.5 * num_slices) + 1) + 1
    # Randomly choose that number of slices
    flattened_indices = rng.choice(num_slices, num_updates, replace=False)
    # Unflatten the indices to match the shape of the tensor
    indices = np.array(np.unravel_index(flattened_indices,
                                        input_shape[0:index_depth])).reshape([index_depth,
                                                                              num_updates]).T

    # Sample slices for that many updates
    updates = rng.choice(max_val, [num_updates, *input_shape[index_depth::]])
    # Sample input tensor
    inputs = rng.choice(max_val, input_shape)

    # Calculate output using tensorflow op
    output = scatter_nd_op(inputs, indices, updates)

    # Return dictionary with the inputs, indices and updates used to produce
    # the result.
    return dict(input=inputs, indices=indices, updates=updates, output=output)


TEST_CASE_TPL = "{test_type}"
TEST_NAME_TPL = "{in_s[0]}x{in_s[1]}x{in_s[2]}x{in_s[3]}_{slice_type}"
IN_SHAPE_INIT_TPL = "{{{{ {0[0]}, {0[1]}, {0[2]}, {0[3]} }}}}"

OPERATOR_MAP = {
    'scatter_nd_assign': 'scatter_nd::Assign',
    'scatter_nd_add': 'scatter_nd::Add',
}


def get_result(test_case, test_params):
    func = get_results
    return func(
        max_val=10,
        index_depth=test_params.index_depth,
        scatter_nd_op=TENSORFLOW_OPS_MAP[test_case.test_type],
        input_shape=test_params.in_shape)


def get_test_lines(test_case, test_params):
    """
    Create a list of strings corresponding to the lines in a single test case.

    Uses TensorFlow to compute the expected results for the given parameters,
    and provides the code to call the test fixture to run the test.
    """
    test_data = get_result(test_case, test_params)
    camel_case_type = helpers.to_camel_case(test_case.test_type)
    test_case_name = TEST_CASE_TPL.format(test_type=camel_case_type)
    test_name = TEST_NAME_TPL.format(
        in_s=test_params.in_shape, slice_type=INDEX_DEPTH_MAP[test_params.index_depth - 1])
    in_shape_init = IN_SHAPE_INIT_TPL.format(test_params.in_shape)
    ind_shape = test_data["indices"].shape
    test_lines = [
        "TYPED_TEST({}, {}) {{".format(test_case_name, test_name),
        "  using DataType = typename TestFixture::DataType;",
        "  const std::vector<DataType> exp_out = {};".format(
            helpers.format_tensor(test_data["output"])),
        "  const std::array<int, 4> in_shape = {};".format(in_shape_init),
        "  const std::array<int, 2> ind_shape = {" + str(ind_shape[0]) + ", " + str(ind_shape[1]) + "};",
        "  const auto params = getScatterNDParams(in_shape, ind_shape);",
        "  const std::vector<DataType> input = {};".format(
            helpers.format_tensor(test_data["input"])),
        "  const std::vector<int> indices = {};".format(
            helpers.format_tensor(test_data["indices"])),
        "  const std::vector<DataType> updates = {};".format(
            helpers.format_tensor(test_data["updates"])),
        "  this->test_scatter_nd(input, indices, updates, exp_out, params);",
        "}",
    ]
    return test_lines


def test_params_for_test_case(test_case):
    "Test params generator for all different tests in a given test case."
    for in_shape in itertools.product(BATCHES, ROWS, COLUMNS, CHANNELS):
        for index_depth in INDEX_DEPTHS:
            yield TestParams(in_shape=in_shape,
                             index_depth=index_depth)


def output_for_test_case(test_case):
    """
    Create a list of strings corresponding to separate lines in the full test
    case. The output contains headers, includes, setup and all the tests for
    the test case.
    """
    scriptname = os.path.basename(__file__)
    camel_case_type = helpers.to_camel_case(test_case.test_type)
    test_case_name = TEST_CASE_TPL.format(test_type=camel_case_type)
    output = [
        helpers.get_license(),
        helpers.get_dont_modify_comment(scriptname=scriptname),
        INCLUDES,
        TYPED_TEST_CASE_DECL_TPL.format(
            test_case=test_case_name,
            operator=OPERATOR_MAP[test_case.test_type])]

    for test_params in test_params_for_test_case(test_case):
        output.extend(get_test_lines(test_case, test_params))
    output.append("\n")
    return output


FILENAME_TPL = "scatter_nd/{test_type}.cc"


def get_test_case_filename(test_case):
    "Get filename for test case."
    return FILENAME_TPL.format(test_type=test_case.test_type)


def test_cases():
    "Test case generator giving all possible test cases."
    for test_type in TEST_TYPES:
        yield TestCaseParams(test_type=test_type)


def generate_scatter_nd_tests():
    np.set_printoptions(suppress=True, threshold=1000000, linewidth=1000000)
    test_dir = helpers.get_test_directory()
    os.chdir(test_dir)
    for test_case in test_cases():
        filename = get_test_case_filename(test_case)
        output = output_for_test_case(test_case)

        with open(filename, 'w') as f:
            f.write('\n'.join(output))
        print("File '{}' written".format(filename))


if __name__ == "__main__":
    generate_scatter_nd_tests()
