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
IN_SIZES = [1, 8, 9]
DIRECTIONS = ['Forward', 'Gradient']
OPERATIONS = ['Training', 'Frozen']

INCLUDES = r"""
#include <gtest/gtest.h>

#include "portdnn/data_format.h"

#include "portdnn/batchnorm/direction.h"
#include "portdnn/batchnorm/params.h"

#include "test/batchnorm/batchnorm_fixture.h"
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
template <typename Triple>
using {test_case} = BatchNormFixture<Triple, batchnorm::{direction}>;
TYPED_TEST_CASE({test_case}, GTestTypeTriples);"""

TestCaseParams = namedtuple('TestCaseParams', ['direction', 'operation'])
TestParams = namedtuple(
    'TestParams', ['in_shape', 'is_training', 'momentum', 'epsilon'])


def compute_gradients(grad_y,
                      x,
                      scale,
                      max_pop_mean_val,
                      max_pop_var_val,
                      input_shape,
                      epsilon,
                      is_training):
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
        mean_grad_y = np.mean(grad_y, axis=(0, 1, 2))
        mean_x = np.mean(x, axis=(0, 1, 2))
        var_x = np.var(x, axis=(0, 1, 2))
        grad_y_offset = grad_y - mean_grad_y
        x_offset = x - mean_x
        mean = np.mean(grad_y * x_offset, axis=(0, 1, 2))
        grad_x = scale * np.reciprocal(np.sqrt(var_x + epsilon)) * (
            grad_y_offset - np.reciprocal(var_x + epsilon) * mean * x_offset)
        grad_scale = np.reciprocal(np.sqrt(var_x + epsilon)) * np.sum(
            grad_y * x_offset, axis=(0, 1, 2))
        grad_offset = np.sum(grad_y, axis=(0, 1, 2))
        return grad_x, grad_scale, grad_offset
    else:
        channel_size = input_shape[-1]
        pop_mean = np.array(
            helpers.get_tensor_data(
                channel_size,
                max_pop_mean_val))
        pop_var = np.array(
            helpers.get_tensor_data(
                channel_size,
                max_pop_var_val))
        grad_offset = np.sum(grad_y, axis=(0, 1, 2))
        var_rsqrt = np.reciprocal(np.sqrt(pop_var + epsilon))
        grad_scale = np.sum(
            grad_y * (x - pop_mean) * var_rsqrt, axis=(0, 1, 2))
        grad_x = grad_y * scale * var_rsqrt
        return grad_x, grad_scale, grad_offset


def compute_batchnorm_grad(
        max_input_val,
        max_gradient_val,
        max_gamma_val,
        max_pop_mean_val,
        max_pop_var_val,
        input_shape,
        epsilon,
        is_training):
    """
    Compute gradient batchnorm.

    Will create an input tensor of the required size filled with values 1, 2,
    3... and use these to compute the batchnorm op. Returns the computed values
    in a numpy array.
    """
    total_inp_size = np.product(input_shape)
    channel_size = input_shape[-1]

    input_vals = helpers.get_tensor_data(total_inp_size, max_input_val)
    gradient_vals = helpers.get_tensor_data(total_inp_size, max_gradient_val)
    gamma_vals = helpers.get_tensor_data(channel_size, max_gamma_val)

    inp_tensor = tf.constant(input_vals,
                             shape=input_shape,
                             dtype=np.float64)
    gradient_tensor = tf.constant(gradient_vals,
                                  shape=input_shape,
                                  dtype=np.float64)
    gamma_tensor = tf.constant(gamma_vals,
                               shape=[channel_size],
                               dtype=np.float64)

    grad_x, grad_scale, grad_offset = compute_gradients(
        gradient_tensor, inp_tensor, gamma_tensor, max_pop_mean_val,
        max_pop_var_val, input_shape, epsilon, is_training)
    return grad_x, grad_scale, grad_offset


def compute_batchnorm_fwd(
        max_input_val,
        max_beta_val,
        max_gamma_val,
        max_input_mean_val,
        max_input_var_val,
        input_shape,
        momentum,
        epsilon,
        is_training):
    """
    Compute forward batchnorm.

    Will initialize inputs with values from 1 to max_input_* and use these to compute the batchnorm op.
    Returns the computed values in a numpy array.
    """
    total_inp_size = np.product(input_shape)
    channel_size = input_shape[-1]

    input_vals = helpers.get_tensor_data(total_inp_size, max_input_val)
    beta_vals = helpers.get_tensor_data(channel_size, max_beta_val)
    gamma_vals = helpers.get_tensor_data(channel_size, max_gamma_val)
    input_mean_vals = helpers.get_tensor_data(channel_size, max_input_mean_val)
    input_var_vals = helpers.get_tensor_data(channel_size, max_input_var_val)

    inp_tensor = tf.constant(input_vals,
                             shape=input_shape,
                             dtype=np.float64)
    beta_tensor = tf.constant(beta_vals,
                              shape=[channel_size],
                              dtype=np.float64)
    gamma_tensor = tf.constant(gamma_vals,
                               shape=[channel_size],
                               dtype=np.float64)
    input_mean_tensor = tf.constant(input_mean_vals,
                                    shape=[channel_size],
                                    dtype=np.float64)
    input_var_tensor = tf.constant(input_var_vals,
                                   shape=[channel_size],
                                   dtype=np.float64)

    output = tf.nn.batch_normalization(
        inp_tensor,
        input_mean_tensor,
        input_var_tensor,
        beta_tensor,
        gamma_tensor,
        epsilon)

    if is_training:
        running_mean = input_mean_tensor * momentum + \
            tf.math.reduce_mean(inp_tensor, axis=[0, 1, 2]) * (1 - momentum)
        running_variance = input_var_tensor * momentum + \
            tf.math.reduce_variance(inp_tensor, axis=[0, 1, 2]) * (1 - momentum)
    else:
        running_mean = []
        running_variance = []

    return output, running_mean, running_variance


TEST_CASE_TPL = "Batchnorm{direction}{operation}"
TEST_NAME_TPL = "{in_s[0]}x{in_s[1]}x{in_s[2]}x{in_s[3]}"
IN_SHAPE_INIT_TPL = "{{{{ {0[0]}, {0[1]}, {0[2]}, {0[3]} }}}}"


def get_forward_result(max_input_val, test_params):
    max_beta_val = 4
    max_gamma_val = 5
    max_input_mean_val = 6
    max_input_var_val = 7
    output, running_mean, running_variance = compute_batchnorm_fwd(
        max_input_val, max_beta_val, max_gamma_val, max_input_mean_val,
        max_input_var_val, test_params.in_shape, test_params.momentum,
        test_params.epsilon, test_params.is_training)
    return (output, running_mean, running_variance, max_input_val,
            max_beta_val, max_gamma_val, max_input_mean_val, max_input_var_val)


def get_gradient_result(max_input_val, test_params):
    max_gradient_val = 4
    max_gamma_val = 5
    max_pop_mean_val = 6
    max_pop_var_val = 7
    grad_x, grad_scale, grad_offset = compute_batchnorm_grad(
        max_input_val, max_gradient_val, max_gamma_val, max_pop_mean_val,
        max_pop_var_val, test_params.in_shape, test_params.epsilon,
        test_params.is_training)
    return (grad_x, grad_scale, grad_offset, max_input_val, max_gradient_val,
            max_gamma_val, max_pop_mean_val, max_pop_var_val)


def get_test_lines(test_case, test_params):
    """
    Create a list of strings corresponding to the lines in a single test case.

    Uses TensorFlow to compute the expected results for the given parameters,
    and provides the code to call the test fixture to run the test.
    """
    test_case_name = TEST_CASE_TPL.format(
        direction=test_case.direction, operation=test_case.operation)
    test_name = TEST_NAME_TPL.format(in_s=test_params.in_shape)
    in_shape_init = IN_SHAPE_INIT_TPL.format(test_params.in_shape)
    if test_case.direction == 'Forward':
        (output, running_mean, running_variance, max_input_val, max_beta_val,
         max_gamma_val, max_input_mean_val, max_input_var_val), _ = \
            helpers.get_result_and_size(get_forward_result, max_input_val=10,
                                        test_params=test_params)
        test_lines = [
            "TYPED_TEST({}, {}) {{".format(
                test_case_name,
                test_name),
            "  using DataType = typename TestFixture::DataType;",
            "  const std::vector<DataType> exp_running_mean = {};".format(
                helpers.format_tensor(running_mean)),
            "  const std::vector<DataType> exp_running_var = {};".format(
                helpers.format_tensor(running_variance)),
            "  const std::vector<DataType> exp_out = {};".format(
                helpers.format_tensor(output)),
            "  const std::array<int, 4> in_shape = {};".format(in_shape_init),
            "  const bool is_training = {};".format(helpers.to_lower_case_str(test_params.is_training)),
            "  const float momentum = {};".format(test_params.momentum),
            "  const float epsilon = {};".format(test_params.epsilon),
            "  const auto params = getBatchNormParams(in_shape, is_training, momentum, epsilon);",
            "  const DataType max_input_val = {:.1f};".format(max_input_val),
            "  const DataType max_beta_val = {:.1f};".format(max_beta_val),
            "  const DataType max_gamma_val = {:.1f};".format(max_gamma_val),
            "  const DataType max_input_mean_val = {:.1f};".format(
                max_input_mean_val),
            "  const DataType max_input_var_val = {:.1f};".format(
                max_input_var_val),
            "  this->test_batchnorm(exp_running_mean, exp_running_var, ",
            "                       exp_out, params, max_input_val, ",
            "                       max_beta_val, max_gamma_val, ",
            "                       max_input_mean_val, max_input_var_val);",
            "}",
        ]
        return test_lines
    else:
        (grad_x, grad_scale, grad_offset, max_input_val, max_gradient_val,
         max_gamma_val, max_pop_mean_val, max_pop_var_val), _ = \
            helpers.get_result_and_size(get_gradient_result, max_input_val=10,
                                        test_params=test_params)
        test_lines = [
            "TYPED_TEST({}, {}) {{".format(test_case_name, test_name),
            "  using DataType = typename TestFixture::DataType;",
            "  const std::vector<DataType> exp_grad = {};".format(
                helpers.format_tensor(grad_x)),
            "  const std::vector<DataType> beta_grad = {};".format(
                helpers.format_tensor(grad_offset)),
            "  const std::vector<DataType> gamma_grad = {};".format(
                helpers.format_tensor(grad_scale)),
            "  const std::array<int, 4> in_shape = {};".format(in_shape_init),
            "  const bool is_training = {};".format(helpers.to_lower_case_str(test_params.is_training)),
            "  const float momentum = {};".format(test_params.momentum),
            "  const float epsilon = {};".format(test_params.epsilon),
            "  const auto params = getBatchNormParams(in_shape, is_training, momentum, epsilon);",
            "  const DataType max_input_val = {:.1f};".format(max_input_val),
            "  const DataType max_gradient_val = {:.1f};".format(max_gradient_val),
            "  const DataType max_gamma_val = {:.1f};".format(max_gamma_val),
            "  const DataType max_pop_mean_val = {:.1f};".format(max_pop_mean_val),
            "  const DataType max_pop_var_val = {:.1f};".format(max_pop_var_val),
            "  this->test_batchnorm(exp_grad, beta_grad, gamma_grad, params, ",
            "                       max_input_val, max_gradient_val, ",
            "                       max_gamma_val, max_pop_mean_val, ",
            "                       max_pop_var_val);",
            "}",
        ]
        return test_lines


def test_params_for_test_case(test_case):
    "Test params generator for all different tests in a given test case."
    for in_shape in itertools.product(BATCHES, IN_SIZES, IN_SIZES, CHANNELS):
        yield TestParams(in_shape=in_shape, is_training=test_case.operation == 'Training', momentum=0.99, epsilon=0.001)


def output_for_test_case(test_case):
    """
    Create a list of strings corresponding to separate lines in the full test
    case. The output contains headers, includes, setup and all the tests for
    the test case.
    """
    scriptname = os.path.basename(__file__)
    test_case_name = TEST_CASE_TPL.format(
        direction=test_case.direction, operation=test_case.operation)
    output = [
        helpers.get_license(),
        helpers.get_dont_modify_comment(scriptname=scriptname),
        INCLUDES,
        TEST_TYPES_TPL,
        TYPED_TEST_CASE_DECL_TPL.format(
            test_case=test_case_name,
            direction=test_case.direction),
    ]

    for test_params in test_params_for_test_case(test_case):
        output.extend(get_test_lines(test_case, test_params))
    output.append("\n")
    return output


FILENAME_TPL = "batchnorm/batchnorm_{direction}_{operation}.cc"


def get_test_case_filename(test_case):
    "Get filename for test case."
    return FILENAME_TPL.format(
        direction=helpers.to_lower_case_str(
            test_case.direction), operation=helpers.to_lower_case_str(
            test_case.operation))


def test_cases():
    "Test case generator giving all possible test cases."
    for direction, operation in itertools.product(DIRECTIONS, OPERATIONS):
        yield TestCaseParams(direction=direction, operation=operation)


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
