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

import itertools
import os
from collections import namedtuple

import tensorflow.compat.v1 as tf
import numpy as np

import helpers

BATCHES = [1, 3]
CHANNELS = [1, 5, 8]
IN_SIZES = [1, 8, 9]  # Assumes square inputs in the spatial dimensions.
TEST_TYPES = ['batchnorm']
DIRECTIONS = ['forward', 'gradient']
OPERATIONS = ['Training', 'Frozen']

INCLUDES = r"""
#include <gtest/gtest.h>

#include "sycldnn/data_format.h"

#include "sycldnn/batchnorm/direction.h"
#include "sycldnn/batchnorm/params.h"

#include "test/batchnorm/batchnorm_fixture.h"
#include "test/types/cartesian_product.h"
#include "test/types/kernel_data_types.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"

#include <vector>"""
TYPED_TEST_CASE_DECL_TPL = r"""
using namespace sycldnn; // NOLINT(google-build-using-namespace)
template <typename Pair>
using {test_case} = BatchNormFixture<Pair, {direction}, {operation}>;
TYPED_TEST_CASE({test_case}, GTestTypePairs);"""

TestCaseParams = namedtuple(
    'TestCaseParams', [
        'test_type', 'direction', 'operation'])
TestParams = namedtuple('TestParams', ['in_shape', 'data_format'])


def compute_gradients(grad_y,
                      x,
                      scale,
                      pop_mean,
                      pop_var,
                      epsilon,
                      is_training=True,
                      keepdims=False):
    """Returns the gradients for the 3 inputs of BatchNorm.
    https://github.com/tensorflow/tensorflow/blob/d916f20e1f1897696a19158ac7f5bd8d83e1b857/tensorflow/python/ops/nn_grad.py#L924
    Args:
      grad_y: A `Tensor` of 4 or 5 dimensions for gradient for y.
      x: A `Tensor` of 4 or 5 dimensions for x.
      scale: A `Tensor` of 1 dimension for scaling.
      pop_mean: A `Tensor` of 1 dimension for the population mean. Only used when
        is_training=False.
      pop_var: A `Tensor` of 1 dimension for the population variance. Only used
        when is_training=False.
      epsilon: A small float number added to the variance of x.
      is_training: A bool value to indicate the operation is for training
        (default) or inference.

    Returns:
      A tuple (grad_x, grad_scale, grad_offset), where grad_x is the gradient
      for x, grad_scale the gradient for scale, and grad_offset the gradient
      for offset.
    """
    if is_training:
        mean_grad_y = np.mean(grad_y, axis=(0, 1, 2), keepdims=keepdims)
        mean_x = np.mean(x, axis=(0, 1, 2), keepdims=keepdims)
        var_x = np.var(x, axis=(0, 1, 2), keepdims=keepdims)
        grad_y_offset = grad_y - mean_grad_y
        x_offset = x - mean_x
        mean = np.mean(grad_y * x_offset, axis=(0, 1, 2), keepdims=keepdims)
        grad_x = scale * np.reciprocal(np.sqrt(var_x + epsilon)) * (
            grad_y_offset - np.reciprocal(var_x + epsilon) * mean * x_offset)
        grad_scale = np.reciprocal(np.sqrt(var_x + epsilon)) * np.sum(
            grad_y * x_offset, axis=(0, 1, 2), keepdims=keepdims)
        grad_offset = np.sum(grad_y, axis=(0, 1, 2))
        return grad_x, grad_scale, grad_offset
    else:
        grad_offset = np.sum(grad_y, axis=(0, 1, 2))
        var_rsqrt = np.reciprocal(np.sqrt(pop_var + epsilon))
        grad_scale = np.sum(
            grad_y * (x - pop_mean) * var_rsqrt, axis=(0, 1, 2))
        grad_x = grad_y * scale * var_rsqrt
        return grad_x, grad_scale, grad_offset


def get_gradient_results(max_val, input_shape, is_training):
    """
    Compute gradient batchnorm.

    Will create an input tensor of the required size filled with values 1, 2,
    3... and use these to compute the batchnorm op. Returns the computed values
    in a numpy array.
    """
    total_inp_size = np.product(input_shape)

    input_vals = helpers.get_tensor_data(total_inp_size, max_val)

    inp_tensor = tf.constant(input_vals,
                             shape=input_shape,
                             dtype=np.float64)

    mean = tf.math.reduce_mean(inp_tensor, axis=[0, 1, 2])

    variance = tf.math.reduce_variance(inp_tensor, axis=[0, 1, 2])

    output = tf.nn.batch_normalization(
        inp_tensor, mean, variance, 0., 1., 0.001)

    grad_x, grad_scale, grad_offset = compute_gradients(
        grad_y=np.array(
            input_vals, dtype=np.float64).reshape(input_shape), x=output, scale=np.ones(
            input_shape[3], dtype=np.float64), pop_mean=mean, pop_var=variance, epsilon=0.001, is_training=is_training)
    return grad_x, mean, variance, grad_scale, grad_offset


def get_forward_results(max_val, input_shape):
    """
    Compute forward batchnorm.

    Will create an input tensor of the required size filled with values 1, 2,
    3... and use these to compute the batchnorm op. Returns the computed values
    in a numpy array.
    """
    total_inp_size = np.product(input_shape)

    input_vals = helpers.get_tensor_data(total_inp_size, max_val)

    inp_tensor = tf.constant(input_vals,
                             shape=input_shape,
                             dtype=np.float64)

    mean = tf.math.reduce_mean(inp_tensor, axis=[0, 1, 2])

    variance = tf.math.reduce_variance(inp_tensor, axis=[0, 1, 2])

    output = tf.nn.batch_normalization(
        inp_tensor, mean, variance, 0., 1., 0.001)

    return output, mean, variance


def get_result_function(test_case):
    """
    Get the function which will compute the expected values for the given test case.
    """
    if (test_case.direction == 'gradient'):
        return get_gradient_results
    elif (test_case.direction == 'forward'):
        return get_forward_results
    else:
        raise Exception("Direction provided not recognised")


TEST_CASE_TPL = "{test_type}{direction}{operation}"
TEST_NAME_TPL = "{in_s[0]}x{in_s[1]}x{in_s[2]}x{in_s[3]}"
IN_SHAPE_INIT_TPL = "{{{{ {0[0]}, {0[1]}, {0[2]}, {0[3]} }}}}"


DIRECTION_MAP = {
    'forward': 'batchnorm::Forward',
    'gradient': 'batchnorm::Gradient'
}

OPERATION_MAP = {
    'Training': 'batchnorm::Training',
    'Frozen': 'batchnorm::Frozen'
}


def get_result(test_case, test_params):
    REQUIRED_MAX = 2**24
    max_input_val = max(
        test_params.in_shape[0],
        test_params.in_shape[1],
        test_params.in_shape[2],
        test_params.in_shape[3])
    max_output_val = REQUIRED_MAX + 1
    floor_div = True
    input_shape = test_params.in_shape
    while max_output_val > REQUIRED_MAX:
        if floor_div:
            max_input_val = max_input_val // 2
        else:
            max_input_val /= 2
        func = get_result_function(test_case)
        if test_case.direction == 'forward':
            output, mean, variance = func(max_val=max_input_val,
                                          input_shape=input_shape)
        else:
            if test_case.operation == 'Training':
                output, mean, variance, grad_scale, grad_offset = func(
                    max_val=max_input_val, input_shape=input_shape, is_training=True)
            else:
                output, mean, variance, grad_scale, grad_offset = func(
                    max_val=max_input_val, input_shape=input_shape, is_training=False)
        max_output_val = np.max(output)
    if test_case.direction == 'forward':
        return output, mean, variance, max_input_val
    else:
        return output, mean, variance, max_input_val, grad_scale, grad_offset


def get_test_lines(test_case, test_params):
    """
    Create a list of strings corresponding to the lines in a single test case.

    Uses TensorFlow to compute the expected results for the given parameters,
    and provides the code to call the test fixture to run the test.
    """
    if test_case.direction == 'forward':
        output, mean, variance, max_input_val = get_result(
            test_case, test_params)
    else:
        output, mean, variance, max_input_val, grad_scale, grad_offset = get_result(
            test_case, test_params)
    camel_case_type = helpers.to_camel_case(test_case.test_type)
    test_case_name = TEST_CASE_TPL.format(test_type=camel_case_type,
                                          direction=helpers.to_camel_case(
                                              test_case.direction),
                                          operation=helpers.to_camel_case(
                                              test_case.operation))
    test_name = TEST_NAME_TPL.format(in_s=test_params.in_shape)
    in_shape_init = IN_SHAPE_INIT_TPL.format(test_params.in_shape)
    if test_case.direction == 'forward':
        test_lines = [
            "TYPED_TEST({}, {}) {{".format(
                test_case_name,
                test_name),
            "  using DataType = typename TestFixture::DataType;",
            "  const std::vector<DataType> exp_out = {};".format(
                helpers.format_tensor(output)),
            " const std::vector<DataType> mean = {};".format(
                helpers.format_tensor(mean)),
            " const std::vector<DataType> variance = {};".format(
                helpers.format_tensor(variance)),
            "  const std::array<int, 4> in_shape = {};".format(in_shape_init),
            "  const auto params = getBatchNormParams(in_shape, DataFormat::{});".format(
                test_params.data_format),
            "  const DataType max_input_val = {:.1f};".format(max_input_val),
            "  this->test_batchnorm(exp_out, mean, variance, params, max_input_val);",
            "}",
        ]
        return test_lines
    else:
        test_lines = [
            "TYPED_TEST({}, {}) {{".format(test_case_name, test_name),
            "  using DataType = typename TestFixture::DataType;",
            "  const std::vector<DataType> exp_grad = {};".format(
                helpers.format_tensor(output)),
            "  const std::vector<DataType> mean = {};".format(helpers.format_tensor(mean)),
            "  const std::vector<DataType> variance = {};".format(helpers.format_tensor(variance)),
            "  const std::vector<DataType> grad_scale = {};".format(helpers.format_tensor(grad_scale)),
            "  const std::vector<DataType> grad_offset = {};".format(helpers.format_tensor(grad_offset)),
            "  const std::array<int, 4> in_shape = {};".format(in_shape_init),
            "  const auto params = getBatchNormParams(in_shape, DataFormat::{});".format(test_params.data_format),
            "  const DataType max_input_val = {:.1f};".format(max_input_val),
            "  this->test_batchnorm(exp_grad, mean, variance, grad_scale, grad_offset, params, max_input_val);",
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
    test_case_name = TEST_CASE_TPL.format(
        test_type=camel_case_type, direction=helpers.to_camel_case(
            test_case.direction), operation=helpers.to_camel_case(
            test_case.operation))
    output = [
        helpers.get_license(),
        helpers.get_dont_modify_comment(scriptname=scriptname),
        INCLUDES,
        helpers.get_test_types_tpl(),
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
    for test_type, direction, operation in itertools.product(
            TEST_TYPES, DIRECTIONS, OPERATIONS):
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
