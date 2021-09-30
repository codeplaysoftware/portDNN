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
# Automatically generate the batchnorm test cases using TensorFlow to provide
# the expected values.

from __future__ import print_function

try:
    # With python3 `zip` returns an iterator, however with python2, use
    # `itertools.izip` instead
    import itertools.izip as zip
except ImportError:
    pass

import itertools
import os
from collections import namedtuple

import tensorflow.compat.v1 as tf
from tensorflow.python.framework.ops import get_gradient_function
import numpy as np

import helpers

BATCHES = [1, 3]
CHANNELS = [1, 5, 8]
IN_SIZES = [1, 8, 9]  # Assumes square inputs in the spatial dimensions.
TEST_TYPES = ['batchnorm']
DIRECTIONS = ['forward']
OPERATIONS = ['Training', 'Inference']

INCLUDES = r"""
#include <gtest/gtest.h>

#include "sycldnn/data_format.h"

#include "sycldnn/batchnorm/direction.h"
#include "sycldnn/batchnorm/params.h"

#include "test/batchnorm/batchnorm_fixture.h"
#include "test/types/kernel_data_types.h"

#include <vector>"""
TYPED_TEST_CASE_DECL_TPL = r"""
using namespace sycldnn; // NOLINT(google-build-using-namespace)
template <typename DataType>
using {test_case} = BatchNormFixture<DataType, {direction}, {operation}>;
TYPED_TEST_CASE({test_case}, types::GTestKernelDataTypes);"""

TestCaseParams = namedtuple('TestCaseParams', ['test_type', 'direction', 'operation'])
TestParams = namedtuple('TestParams', ['in_shape', 'data_format'])

TENSORFLOW_OPS_MAP = {
    'batchnorm': tf.nn.batch_normalization,
}

def get_forward_results(max_val, input_shape):
    """
    Construct and run a Tensorflow graph to compute a forward batchnorm op.

    Will create an input tensor of the required size filled with values 1, 2,
    3... and use these to compute the batchnorm op. Returns the computed values
    in a numpy array.
    """
    with tf.Graph().as_default():
        total_inp_size = np.product(input_shape)

        input_vals = helpers.get_tensor_data(total_inp_size, max_val)

        inp_tensor = tf.constant(input_vals,
                                 shape=input_shape,
                                 dtype=np.float32)

        mean = tf.math.reduce_mean(inp_tensor,axis=[0,1,2])

        variance = tf.math.reduce_variance(inp_tensor,axis=[0,1,2])

        output = tf.nn.batch_normalization(inp_tensor, mean, variance, 0., 1., 0.001)
        
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            sess.graph.finalize()
            return sess.run(output), sess.run(mean), sess.run(variance)


def get_result_function(test_case):
    """
    Get the function which will compute the expected values for the given test case.
    """
    return get_forward_results


#TODO dansoutar: fix these and the remainder of the file.
TEST_CASE_TPL = "{test_type}{direction}{operation}"
TEST_NAME_TPL = "{in_s[0]}x{in_s[1]}x{in_s[2]}x{in_s[3]}"
IN_SHAPE_INIT_TPL = "{{{{ {0[0]}, {0[1]}, {0[2]}, {0[3]} }}}}"


DIRECTION_MAP = {
    'forward': 'batchnorm::Forward'
}

OPERATION_MAP = {
    'Training': 'batchnorm::Training',
    'Inference': 'batchnorm::Inference'
}


def get_result(test_case, test_params):
    REQUIRED_MAX = 2**24
    max_input_val=max(test_params.in_shape[0], test_params.in_shape[1], test_params.in_shape[2], test_params.in_shape[3])
    max_output_val = REQUIRED_MAX + 1
    floor_div=True
    input_shape=test_params.in_shape
    while max_output_val > REQUIRED_MAX:
        if floor_div:
            max_input_val = max_input_val // 2
        else:
            max_input_val /= 2
        func = get_result_function(test_case)
        output, mean, variance = func(max_input_val, input_shape)
        max_output_val = np.max(output)
    return output, mean, variance, max_input_val


def get_test_lines(test_case, test_params):
    """
    Create a list of strings corresponding to the lines in a single test case.

    Uses TensorFlow to compute the expected results for the given parameters,
    and provides the code to call the test fixture to run the test.
    """
    channel_idx = -1 if test_params.data_format == 'NHWC' else 1
    output, mean, variance, max_input_val = get_result(test_case, test_params)
    camel_case_type = helpers.to_camel_case(test_case.test_type)
    test_case_name = TEST_CASE_TPL.format(test_type=camel_case_type,
                                          direction=helpers.to_camel_case(
                                              test_case.direction),
                                              operation=helpers.to_camel_case(
                                              test_case.operation))
    test_name = TEST_NAME_TPL.format(in_s=test_params.in_shape)
    in_shape_init = IN_SHAPE_INIT_TPL.format(test_params.in_shape)
    test_lines = [
        "TYPED_TEST({}, {}) {{".format(test_case_name, test_name),
        "  using DataType = typename TestFixture::DataType;",
        "  const std::vector<DataType> exp_out = {};".format(
            helpers.format_tensor(output)),
        " const std::vector<DataType> mean = {};".format(helpers.format_tensor(mean)),
        " const std::vector<DataType> variance = {};".format(helpers.format_tensor(variance)),
        "  const std::array<int, 4> in_shape = {};".format(in_shape_init),
        "  const auto params = getBatchNormParams(in_shape, DataFormat::{});".format(test_params.data_format),
        "  const DataType max_input_val = {:.1f};".format(max_input_val),
        "  this->test_batchnorm(exp_out, mean, variance, params, max_input_val);",
        "}",
    ]
    return test_lines


def test_params_for_test_case(test_case):
    "Test params generator for all different tests in a given test case."
    for in_shape in itertools.product(BATCHES, IN_SIZES, IN_SIZES, CHANNELS):
        yield TestParams(in_shape=in_shape, data_format='NHWC')


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
                                              test_case.direction),
                                              operation=helpers.to_camel_case(test_case.operation))
    output = [
        helpers.get_license(),
        helpers.get_dont_modify_comment(scriptname=scriptname),
        INCLUDES,
        TYPED_TEST_CASE_DECL_TPL.format(
            test_case=test_case_name,
            direction=DIRECTION_MAP[test_case.direction],
            operation=OPERATION_MAP[test_case.operation]),
    ]

    for test_params in test_params_for_test_case(test_case):
        output.extend(get_test_lines(test_case, test_params))
    output.append("\n")
    return output


FILENAME_TPL = "batchnorm/{test_type}_{direction}_{operation}.cc"


def get_test_case_filename(test_case):
    "Get filename for test case."
    return FILENAME_TPL.format(test_type=test_case.test_type,
                               direction=test_case.direction,
                               operation=test_case.operation)


def test_cases():
    "Test case generator giving all possible test cases."
    for test_type, direction, operation in itertools.product(TEST_TYPES, DIRECTIONS, OPERATIONS):
        yield TestCaseParams(test_type=test_type, direction=direction, operation=operation)


def generate_batchnorm_tests():
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
    generate_batchnorm_tests()

