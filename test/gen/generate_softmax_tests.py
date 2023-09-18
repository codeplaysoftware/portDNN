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
# Automatically generate the softmax test cases using TensorFlow to provide
# the expected values.

from __future__ import print_function

import itertools
import os
from collections import namedtuple

import tensorflow as tf
import numpy as np

import helpers

BATCHES = [1, 3]
CHANNELS = [1, 5, 8]
IN_SIZES = [1, 8, 9]
TEST_TYPES = ['softmax']
DIRECTIONS = ['forward', 'grad']

INCLUDES = r"""
#include <gtest/gtest.h>

#include "portdnn/data_format.h"

#include "portdnn/softmax/direction.h"
#include "portdnn/softmax/params.h"

#include "test/softmax/softmax_fixture.h"
#include "test/types/cartesian_product.h"
#include "test/types/data_format_types.h"
#include "test/types/kernel_data_types.h"
#include "test/types/nested_pairs_to_triple.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"

#include <vector>"""

TEST_TYPES_TPL = r"""
using DataTypeList = sycldnn::types::KernelDataTypes;
using Backends = sycldnn::types::AllBackendTypes;
using DataFormats = sycldnn::types::DataFormatTypes;

using TypeBackendPairs =
    sycldnn::types::CartesianProduct<DataTypeList, Backends>::type;
using TypeBackendFormatTriple =
    sycldnn::types::CartesianProduct<TypeBackendPairs, DataFormats>::type;

using TestTriples =
    sycldnn::types::NestedPairsToTriple<TypeBackendFormatTriple>::type;
using GTestTypeTriples = sycldnn::types::ToGTestTypes<TestTriples>::type;
"""

TYPED_TEST_CASE_DECL_TPL = r"""
using namespace sycldnn; // NOLINT(google-build-using-namespace)
template <typename Tripe>
using {test_case} = SoftmaxFixture<Tripe, {direction}>;
TYPED_TEST_CASE({test_case}, GTestTypeTriples);"""

TestCaseParams = namedtuple('TestCaseParams', ['test_type', 'direction'])
TestParams = namedtuple('TestParams', ['in_shape'])


def get_grad_results(max_val, input_shape):
    """
    Compute backprop softmax.

    Will create an input tensor of the required size filled with values 1, 2,
    3... and use these to compute the softmax op. Then, create another tensor
    with the same values to use as the errors for backpropagation.
    Returns the computed values in a numpy array.
    """
    input = helpers.get_variable(input_shape, max_val)

    with tf.GradientTape() as tape:
        output = tf.nn.softmax(input)

    output_shape = output.shape
    error = helpers.get_variable(output_shape, max_val)
    return tape.gradient(output, input, error)


def get_forward_results(max_val, input_shape):
    """
    Compute forward softmax.

    Will create an input tensor of the required size filled with values 1, 2,
    3... and use these to compute the softmax op. Returns the computed values
    in a numpy array.
    """
    total_inp_size = np.product(input_shape)

    input_vals = helpers.get_tensor_data(total_inp_size, max_val)

    inp_tensor = tf.constant(input_vals,
                             shape=input_shape,
                             dtype=np.float64)

    return tf.nn.softmax(inp_tensor)


def get_result_function(test_case):
    """
    Get the function which will compute the expected values for the given test case.
    """
    if (test_case.direction == 'grad'):
        return get_grad_results
    elif (test_case.direction == 'forward'):
        return get_forward_results
    else:
        raise Exception("Direction provided not recognised")


TEST_CASE_TPL = "{test_type}{direction}"
TEST_NAME_TPL = "{in_s[0]}x{in_s[1]}x{in_s[2]}x{in_s[3]}"
IN_SHAPE_INIT_TPL = "{{{{ {0[0]}, {0[1]}, {0[2]}, {0[3]} }}}}"

DIRECTION_MAP = {
    'forward': 'softmax::Forward',
    'grad': 'softmax::Gradient',
}


def get_test_lines(test_case, test_params):
    """
    Create a list of strings corresponding to the lines in a single test case.

    Uses TensorFlow to compute the expected results for the given parameters,
    and provides the code to call the test fixture to run the test.
    """
    output, max_input_val = helpers.get_result_and_size(
        get_result_function(test_case),
        floor_div=True,
        input_shape=test_params.in_shape)
    camel_case_type = helpers.to_camel_case(test_case.test_type)
    test_case_name = TEST_CASE_TPL.format(test_type=camel_case_type,
                                          direction=helpers.to_camel_case(
                                              test_case.direction))
    test_name = TEST_NAME_TPL.format(in_s=test_params.in_shape)
    in_shape_init = IN_SHAPE_INIT_TPL.format(test_params.in_shape)
    test_lines = [
        "TYPED_TEST({}, {}) {{".format(
            test_case_name,
            test_name),
        "  using DataType = typename TestFixture::DataType;",
        "  const std::vector<DataType> exp_out = {};".format(
            helpers.format_tensor(output)),
        "  const std::array<int, 4> in_shape = {};".format(in_shape_init),
        "  const auto params = getSoftmaxParams(in_shape);",
        "  const DataType max_input_val = {:.1f};".format(max_input_val),
        "  this->test_softmax(exp_out, params, max_input_val);",
        "}",
    ]
    return test_lines


def test_params_for_test_case(test_case):
    "Test params generator for all different tests in a given test case."
    for in_shape in itertools.product(BATCHES, IN_SIZES, IN_SIZES, CHANNELS):
        yield TestParams(in_shape=in_shape)


def output_for_test_case(test_case):
    """
    Create a list of strings corresponding to separate lines in the full test
    case. The output contains headers, includes, setup and all the tests for
    the test case.
    """
    scriptname = os.path.basename(__file__)
    camel_case_type = helpers.to_camel_case(test_case.test_type)
    test_case_name = TEST_CASE_TPL.format(test_type=camel_case_type,
                                          direction=helpers.to_camel_case(
                                              test_case.direction))
    output = [
        helpers.get_license(),
        helpers.get_dont_modify_comment(scriptname=scriptname),
        INCLUDES,
        TEST_TYPES_TPL,
        TYPED_TEST_CASE_DECL_TPL.format(
            test_case=test_case_name,
            direction=DIRECTION_MAP[test_case.direction]),
    ]

    for test_params in test_params_for_test_case(test_case):
        output.extend(get_test_lines(test_case, test_params))
    output.append("\n")
    return output


FILENAME_TPL = "softmax/{test_type}_{direction}.cc"


def get_test_case_filename(test_case):
    "Get filename for test case."
    return FILENAME_TPL.format(test_type=test_case.test_type,
                               direction=test_case.direction)


def test_cases():
    "Test case generator giving all possible test cases."
    for test_type, direction in itertools.product(TEST_TYPES, DIRECTIONS):
        yield TestCaseParams(test_type=test_type, direction=direction)


def generate_softmax_tests():
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
    generate_softmax_tests()
