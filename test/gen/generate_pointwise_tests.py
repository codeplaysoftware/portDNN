#!python
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
# Automatically generate the pointwise test cases using TensorFlow to provide
# the expected values.

from __future__ import print_function

import itertools
import os
from collections import namedtuple

import tensorflow as tf
import numpy as np

import helpers

TEST_TYPES = ["relu", "tanh", "log", "floor", "sqrt"]
DIRECTIONS = ["forward", "grad"]

INCLUDES = r"""
#include <gtest/gtest.h>

#include "portdnn/pointwise/direction.h"
#include "portdnn/pointwise/operators.h"

#include "test/pointwise/pointwise_fixture.h"
#include "test/types/cartesian_product.h"
#include "test/types/kernel_data_types.h"
#include "test/types/test_backend_types.h"
"""
TYPED_TEST_SUITE_DECL_TPL = r"""
using namespace sycldnn;  // NOLINT(google-build-using-namespace)

using DataTypeList = sycldnn::types::KernelDataTypes;
using Backends = sycldnn::types::DefaultBackendTypes;

using TypeBackendPairs =
    sycldnn::types::CartesianProduct<DataTypeList, Backends>::type;

using GTestTypePairs = sycldnn::types::ToGTestTypes<TypeBackendPairs>::type;

template <typename DataType>
using {test_case} =
    PointwiseFixture<DataType, {operation}, {direction}>;
TYPED_TEST_SUITE({test_case}, GTestTypePairs);"""

TestCaseParams = namedtuple("TestCaseParams", ["test_type", "direction"])
TestParams = namedtuple("TestParams", ["in_size"])


class Op_Info:
    """
    Class Op_Info is used to track information about operations including
        Name: String of the op name
        tf_op: the tensorflow operation used
        gradient_support: True if tf_op has a gradient
        forward_signed_input: True if tf_op can take signed values and zero
        grad_signed_input: True if the gradient op can take signed values and zero
    """

    def __init__(self, name, tf_op, gradient_support=True,
                 forward_signed_input=True, grad_signed_input=True):
        self.name = name
        self.tf_op = tf_op
        self.gradient_support = gradient_support
        self.forward_signed_input = forward_signed_input
        self.grad_signed_input = grad_signed_input


TENSORFLOW_OPS_MAP = {
    "relu": Op_Info(name="relu",
                    tf_op=tf.nn.relu,
                    gradient_support=True,
                    forward_signed_input=True,
                    grad_signed_input=True),
    "tanh": Op_Info(name="tanh",
                    tf_op=tf.nn.tanh,
                    gradient_support=True,
                    forward_signed_input=True,
                    grad_signed_input=True),
    "log": Op_Info(name="log",
                   tf_op=tf.math.log,
                   gradient_support=True,
                   forward_signed_input=False,
                   grad_signed_input=False),
    "floor": Op_Info(name="floor",
                     tf_op=tf.math.floor,
                     gradient_support=False,
                     forward_signed_input=True),
    "sqrt": Op_Info(name="sqrt",
                    tf_op=tf.math.sqrt,
                    gradient_support=True,
                    forward_signed_input=False,
                    grad_signed_input=False),
}


def get_input_tensor(direction, pointwise_op, in_size, max_val):
    """
    Get the input values to be used when calulating results.
    Returns a tf variable.
    """
    min_val = -max_val if in_size % 2 == 0 else -max_val - 1

    if direction == 'grad':
        if pointwise_op.grad_signed_input:
            input_vals = helpers.get_signed_variable(in_size, min_val, max_val)
            input_func = "iota_initialised_signed_data"
            input_params = "{}".format(in_size)
        else:
            input_vals = helpers.get_variable(in_size, in_size)
            input_func = "iota_initialised_data"
            input_params = "{}, {}".format(in_size, in_size)
    elif direction == 'forward':
        if pointwise_op.forward_signed_input:
            input_vals = helpers.get_signed_tensor_data(in_size,
                                                        min_val=min_val,
                                                        max_val=max_val)
            input_func = "iota_initialised_signed_data"
            input_params = "{}".format(in_size)
        else:
            input_vals = helpers.get_tensor_data(in_size,
                                                 max_val=in_size)
            input_func = "iota_initialised_data"
            input_params = "{}, {}".format(in_size, in_size)
        input_vals = tf.Variable(input_vals, dtype=np.float64)
    else:
        raise Exception("Direction {} invalid".format(direction))

    return input_vals, input_func, input_params


def get_error_tensor(pointwise_op, in_size, output_shape, max_val):
    """
    Get the error values to be used when calulating results.
    Returns a tf variable.
    """
    min_val = -max_val if in_size % 2 == 0 else -max_val - 1
    if pointwise_op.grad_signed_input:
        error_tensor = helpers.get_signed_variable(
            output_shape, min_val, max_val)
    else:
        error_tensor = helpers.get_variable(output_shape, output_shape[0])

    return error_tensor


def get_grad_results(max_val, pointwise_op, in_size):
    """
    Compute a backprop pointwise op.

    Will create an input tensor of the required size filled with values -n, -n+1,
    ..., 0, 1, ..., n-1, n and use these to compute the pointwise op.
    Then, create another tensor with the same values to use as the errors
    for backpropagation.
    Returns the computed values in a numpy array.
    """
    input, _, _ = get_input_tensor(
        direction='grad',
        pointwise_op=pointwise_op,
        in_size=in_size,
        max_val=max_val)

    with tf.GradientTape() as tape:
        output = pointwise_op.tf_op(input)

    error = get_error_tensor(pointwise_op=pointwise_op, in_size=in_size,
                             output_shape=output.shape, max_val=max_val)

    return tape.gradient(output, input, error)


def get_forward_results(max_val, pointwise_op, in_size):
    """
    Compute a forward pointwise op.

    Will create an input tensor of the required size filled with values -n, -n+1,
    ..., 0, 1, ..., n-1, n and use these to compute the pointwise op.
    Returns the computed values in a numpy array.
    """

    inp_tensor, _, _ = get_input_tensor(
        direction='forward',
        pointwise_op=pointwise_op,
        in_size=in_size,
        max_val=max_val)

    return pointwise_op.tf_op(inp_tensor)


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
TEST_NAME_TPL = "Shape_{in_s}x1"

OPERATOR_MAP = {
    'relu': 'pointwise::Relu',
    'tanh': 'pointwise::Tanh',
    'log': 'pointwise::Log',
    'floor': 'pointwise::Floor',
    'sqrt': 'pointwise::Sqrt',
}

DIRECTION_MAP = {
    'forward': 'pointwise::Forward',
    'grad': 'pointwise::Gradient',
}


def get_result(test_case, test_params):
    output, max_input_val = helpers.get_result_and_size(
        get_result_function(test_case),
        max_input_val=test_params.in_size,
        floor_div=True,
        pointwise_op=TENSORFLOW_OPS_MAP[test_case.test_type],
        in_size=test_params.in_size)
    return output, max_input_val


def get_test_lines(test_case, test_params):
    """
    Create a list of strings corresponding to the lines in a single test case.

    Uses TensorFlow to compute the expected results for the given parameters,
    and provides the code to call the test fixture to run the test.
    """
    output, max_input_val = get_result(test_case, test_params)
    pointwise_op = TENSORFLOW_OPS_MAP[test_case.test_type]
    _, input_func, input_params = get_input_tensor(
        direction=test_case.direction,
        pointwise_op=pointwise_op,
        in_size=test_params.in_size,
        max_val=max_input_val)
    camel_case_type = helpers.to_camel_case(test_case.test_type)
    test_case_name = TEST_CASE_TPL.format(test_type=camel_case_type,
                                          direction=helpers.to_camel_case(
                                              test_case.direction))
    test_name = TEST_NAME_TPL.format(in_s=test_params.in_size)
    test_lines = [
        "TYPED_TEST({}, {}) {{".format(test_case_name, test_name),
        "  using DataType = typename TestFixture::DataType;",
        "  const std::vector<DataType> input = {}<DataType>({});".format(
            input_func, input_params),
        "  const std::vector<DataType> exp_out = {};".format(
            helpers.format_tensor(output)),
        "  this->test_pointwise(input, exp_out);",
        "}",
    ]
    return test_lines


def test_params_for_test_case(test_case):
    "Test params generator for all different tests in a given test case."
    in_sizes = [1, 8, 9, 10]
    for size in in_sizes:
        yield TestParams(in_size=size)


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
        TYPED_TEST_SUITE_DECL_TPL.format(
            test_case=test_case_name,
            operation=OPERATOR_MAP[test_case.test_type],
            direction=DIRECTION_MAP[test_case.direction]),
    ]

    for test_params in test_params_for_test_case(test_case):
        output.extend(get_test_lines(test_case, test_params))
    output.append("\n")
    return output


FILENAME_TPL = "pointwise/{test_type}_{direction}.cc"


def get_test_case_filename(test_case):
    "Get filename for test case."
    return FILENAME_TPL.format(test_type=test_case.test_type,
                               direction=test_case.direction)


def test_cases():
    "Test case generator giving all possible test cases."
    for test_type, direction in itertools.product(TEST_TYPES, DIRECTIONS):
        if direction == 'forward' or TENSORFLOW_OPS_MAP[test_type].gradient_support:
            yield TestCaseParams(test_type=test_type, direction=direction)


def generate_pointwise_tests():
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
    generate_pointwise_tests()
