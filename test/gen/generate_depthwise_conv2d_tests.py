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
# Automatically generate the convolution test cases using TensorFlow to provide
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

import tensorflow as tf
import numpy as np

import helpers

WINDOW_LIST = [1, 1, 3, 3, 5, 5, 7, 7, 11, 11]
STRIDE_LIST = [1, 2, 1, 2, 1, 2, 1, 4, 1, 4]
BATCHES = [1, 3]
CHANNELS = [1, 2, 4]
MULTIPLIERS = [1, 2, 4]
PADDING_VALUES = ["SAME", "VALID"]
TEST_TYPES = ["forward", "input_backprop", "filter_backprop"]

INCLUDES = r"""
#include <gtest/gtest.h>

#include "portdnn/padding_mode.h"

#include "test/types/cartesian_product.h"
#include "test/types/kernel_data_types.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"

#include "test/depthwise_conv2d/window_stride_fixture.h"

#include <array>
#include <vector>"""
DATA_TYPES = r"""
using DataTypeList = sycldnn::types::KernelDataTypes;
using Backends = sycldnn::types::DefaultBackendTypes;

using SNNTypePairs =
    sycldnn::types::CartesianProduct<DataTypeList, Backends>::type;
using GTestTypePairs = sycldnn::types::ToGTestTypes<SNNTypePairs>::type;
"""
TYPED_TEST_SUITE_DECL_TPL = r"""
template <typename Pair>
using {test_case} =
    sycldnn::depthwise_conv2d::WindowStrideTest<Pair, {window}, {stride}>;
TYPED_TEST_SUITE({test_case}, GTestTypePairs);"""

TestCaseParams = namedtuple('TestCaseParams',
                            ['test_type', 'window', 'stride'])
TestParams = namedtuple('TestParams', ['in_shape', 'multiplier', 'padding'])


def get_forward_conv_results(max_val, input_shape, filter_shape, stride_shape,
                             padding):
    """
    Compute depthwise forward convolution.

    Will create input tensors of the required size filled with values 1, 2,
    3... and use these to compute the convolution for the forward pass.
    Returns the computed values in a numpy array.
    """
    total_inp_size = np.product(input_shape)
    total_fil_size = np.product(filter_shape)

    input_vals = helpers.get_tensor_data(total_inp_size, max_val)
    filter_vals = helpers.get_tensor_data(total_fil_size, max_val)

    inp_tensor = tf.constant(input_vals,
                             shape=input_shape,
                             dtype=np.float64)
    fil_tensor = tf.constant(filter_vals,
                             shape=filter_shape,
                             dtype=np.float64)
    return tf.nn.depthwise_conv2d(inp_tensor,
                                  fil_tensor,
                                  strides=stride_shape,
                                  padding=padding,
                                  data_format="NHWC")


def get_input_backprop_conv_results(max_val, input_shape, filter_shape,
                                    stride_shape, padding):
    """
    Compute depthwise input backprop convolution.

    Will create input tensors of the required size filled with values 1, 2,
    3... and use these to compute the convolution for the input backprop pass.
    Returns the computed values in a numpy array.
    """
    total_fil_size = np.product(filter_shape)
    filter_vals = helpers.get_tensor_data(total_fil_size, max_val)

    inp_tensor = tf.constant(0, shape=input_shape, dtype=np.float64)
    fil_tensor = tf.constant(filter_vals,
                             shape=filter_shape,
                             dtype=np.float64)
    output = tf.nn.depthwise_conv2d(inp_tensor,
                                    fil_tensor,
                                    strides=stride_shape,
                                    padding=padding,
                                    data_format="NHWC")

    output_shape = output.shape
    total_out_size = np.product(output_shape)
    output_vals = helpers.get_tensor_data(total_out_size, max_val)
    out_tensor = tf.constant(output_vals,
                             shape=output_shape,
                             dtype=np.float64)

    inp_size_tensor = tf.constant(input_shape, shape=[len(input_shape)])
    return tf.nn.depthwise_conv2d_backprop_input(
        inp_size_tensor,
        fil_tensor,
        out_tensor,
        strides=stride_shape,
        padding=padding,
        data_format="NHWC")


def get_filter_backprop_conv_results(max_val, input_shape, filter_shape,
                                     stride_shape, padding):
    """
    Compute depthwise filter backprop convolution.

    Will create input tensors of the required size filled with values 1, 2,
    3... and use these to compute the convolution for the filter backprop pass.
    Returns the computed values in a numpy array.
    """
    total_inp_size = np.product(input_shape)
    input_vals = helpers.get_tensor_data(total_inp_size, max_val)

    inp_tensor = tf.constant(input_vals,
                             shape=input_shape,
                             dtype=np.float64)
    fil_tensor = tf.constant(0, shape=filter_shape, dtype=np.float64)
    output = tf.nn.depthwise_conv2d(inp_tensor,
                                    fil_tensor,
                                    strides=stride_shape,
                                    padding=padding,
                                    data_format="NHWC")

    output_shape = output.shape
    total_out_size = np.product(output_shape)
    output_vals = helpers.get_tensor_data(total_out_size, max_val)
    out_tensor = tf.constant(output_vals,
                             shape=output_shape,
                             dtype=np.float64)

    fil_size_tensor = tf.constant(filter_shape, shape=[len(filter_shape)])
    return tf.nn.depthwise_conv2d_backprop_filter(
        inp_tensor,
        fil_size_tensor,
        out_tensor,
        strides=stride_shape,
        padding=padding,
        data_format="NHWC")


def get_conv_fn(test_type):
    """
    Get the function which computes the convolution corresponding to the test type.
    """
    if test_type == "forward":
        return get_forward_conv_results
    elif test_type == "input_backprop":
        return get_input_backprop_conv_results
    elif test_type == "filter_backprop":
        return get_filter_backprop_conv_results
    else:
        raise ValueError("Unknown test type requested.")


def get_result_and_size(test_case, test_params):
    """
    Get the result of the specified convolution and max input value.

    Ensures that the resulting values are less than the REQUIRED_MAX, and if
    not will adjust the maximum value to allow in the input tensors.
    """
    conv_fn = get_conv_fn(test_case.test_type)
    filter_shape = [
        test_case.window, test_case.window, test_params.in_shape[-1],
        test_params.multiplier
    ]
    stride_shape = [1, test_case.stride, test_case.stride, 1]
    return helpers.get_result_and_size(conv_fn,
                                       input_shape=test_params.in_shape,
                                       filter_shape=filter_shape,
                                       stride_shape=stride_shape,
                                       padding=test_params.padding)


TEST_CASE_TPL = "{test_type}Window{window}Stride{stride}"
TEST_NAME_TPL = "{padding}{in_s[0]}x{in_s[1]}x{in_s[2]}x{in_s[3]}x{multiplier}"
IN_SHAPE_INIT_TPL = "{{{{ {0[0]}, {0[1]}, {0[2]}, {0[3]} }}}}"


def get_test_lines(test_case, test_params):
    """
    Create a list of strings corresponding to the lines in a single test case.

    Uses TensorFlow to compute the expected results for the given parameters,
    and provides the code to call the test fixture to run the test.
    """
    output, max_input_val = get_result_and_size(test_case, test_params)
    camel_case_type = helpers.to_camel_case(test_case.test_type)
    test_case_name = TEST_CASE_TPL.format(test_type=camel_case_type,
                                          window=test_case.window,
                                          stride=test_case.stride)
    test_name = TEST_NAME_TPL.format(padding=test_params.padding,
                                     in_s=test_params.in_shape,
                                     multiplier=test_params.multiplier)
    in_shape_init = IN_SHAPE_INIT_TPL.format(test_params.in_shape)
    test_lines = [
        "TYPED_TEST({}, {}) {{".format(test_case_name, test_name),
        "  using DataType = typename TestFixture::DataType;",
        "  const std::vector<DataType> exp_out = {};".format(
            helpers.format_tensor(output)),
        "  const std::array<int, 4> in_shape = {};".format(in_shape_init),
        "  const int multiplier = {};".format(test_params.multiplier),
        "  const auto padding = sycldnn::PaddingMode::{};".format(
            test_params.padding),
        "  const DataType max_input_val = {:.1f};".format(max_input_val),
        "  this->run_{}_test(exp_out, in_shape, multiplier, padding, max_input_val);"
        .format(test_case.test_type),
        "}",
    ]
    return test_lines


def get_input_sizes(test_case):
    """
    Want to test with sizes that are:
        a) Divisible by 4
        b) Divisible by 2 but not 4
        c) Not Divisible by 2
    And we also require the sizes to be large enough that there are at least
    two entries in the output tensor, so the minimum size is (window + stride)
    and the other sizes need to be calculated to ensure that the above criteria
    are satisfied.
    """
    start = test_case.window + test_case.stride
    if start % 2 == 1:
        return [start, start + 1, start + 3]
    else:
        return [start, start + 1, start + 2]


def test_params_for_test_case(test_case):
    "Test params generator for all different tests in a given test case."
    in_sizes = get_input_sizes(test_case)
    for in_shape in itertools.product(BATCHES, in_sizes, in_sizes, CHANNELS):
        for multiplier, padding in itertools.product(MULTIPLIERS,
                                                     PADDING_VALUES):
            yield TestParams(in_shape=in_shape,
                             multiplier=multiplier,
                             padding=padding)


def output_for_test_case(test_case):
    """
    Create a list of strings corresponding to separate lines in the full test
    case. The output contains headers, includes, setup and all the tests for
    the test case.
    """
    scriptname = os.path.basename(__file__)
    camel_case_type = helpers.to_camel_case(test_case.test_type)
    test_case_name = TEST_CASE_TPL.format(test_type=camel_case_type,
                                          window=test_case.window,
                                          stride=test_case.stride)
    output = [
        helpers.get_license(),
        helpers.get_dont_modify_comment(scriptname=scriptname), INCLUDES,
        DATA_TYPES,
        TYPED_TEST_SUITE_DECL_TPL.format(test_case=test_case_name,
                                         window=test_case.window,
                                         stride=test_case.stride)
    ]
    for test_params in test_params_for_test_case(test_case):
        output.extend(get_test_lines(test_case, test_params))
    return output


FILENAME_TPL = "depthwise_conv2d/{test_type}_window{window}_stride{stride}.cc"


def get_test_case_filename(test_case):
    "Get filename for test case."
    return FILENAME_TPL.format(test_type=test_case.test_type,
                               window=test_case.window,
                               stride=test_case.stride)


def test_cases():
    "Test case generator giving all possible test cases."
    for window, stride in zip(WINDOW_LIST, STRIDE_LIST):
        for test_type in TEST_TYPES:
            yield TestCaseParams(test_type=test_type,
                                 window=window,
                                 stride=stride)


def generate_depthwise_conv2d_tests():
    np.set_printoptions(suppress=True,
                        precision=10,
                        threshold=1000000,
                        linewidth=1000000)
    test_dir = helpers.get_test_directory()
    os.chdir(test_dir)
    for test_case in test_cases():
        filename = get_test_case_filename(test_case)
        output = output_for_test_case(test_case)
        with open(filename, 'w') as f:
            f.write('\n'.join(output))
        print("File '{}' written".format(filename))


if __name__ == "__main__":
    generate_depthwise_conv2d_tests()
