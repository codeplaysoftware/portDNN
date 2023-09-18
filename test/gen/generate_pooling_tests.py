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
# Automatically generate the pooling test cases using TensorFlow to provide
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

WINDOW_LIST = [1, 3, 3, 5, 5, 7, 7, 11, 11]
STRIDE_LIST = [1, 1, 2, 1, 2, 1, 4, 1, 4]
BATCHES = [1, 3]
CHANNELS = [1, 2, 4]
PADDING_VALUES = ["SAME", "VALID"]
TEST_TYPES = ["maxwithnan", "max", "avg"]
DIRECTIONS = ['forward', 'grad']
DATA_LAYOUT = ["NHWC", "NCHW"]

INCLUDES = r"""
#include <gtest/gtest.h>

#include "portdnn/padding_mode.h"

#include "portdnn/pooling/operators.h"

#include "test/types/cartesian_product.h"
#include "test/types/data_format_types.h"
#include "test/types/kernel_data_types.h"
#include "test/types/test_backend_types.h"
#include "test/types/nested_pairs_to_triple.h"
#include "test/types/to_gtest_types.h"

#include "test/pooling/pooling_fixture.h"

#include <array>
#include <vector>"""
DATA_TYPES = r"""
using namespace sycldnn; // NOLINT(google-build-using-namespace)
using DataTypeList = sycldnn::types::KernelDataTypes;
using DataFormatList = sycldnn::types::DataFormatTypes;
using BackendList = sycldnn::types::DefaultBackendTypes;

using SNNTypePairs =
    sycldnn::types::CartesianProduct<DataTypeList, DataFormatList>::type;
using SNNTypeBackendPairs =
    sycldnn::types::CartesianProduct<SNNTypePairs, BackendList>::type;
using TestTriples =
    sycldnn::types::NestedPairsToTriple<SNNTypeBackendPairs>::type;
using GTestTypeTriples = sycldnn::types::ToGTestTypes<TestTriples>::type;"""
TYPED_TEST_SUITE_DECL_TPL = r"""
template <typename Triple>
using {test_case} =
    PoolingFixture<typename Triple::FirstType, typename Triple::SecondType,
                   typename Triple::ThirdType, {operation}, {direction}>;
TYPED_TEST_SUITE({test_case}, GTestTypeTriples);"""

TestCaseParams = namedtuple(
    'TestCaseParams',
    ['test_type', 'direction', 'window', 'stride', 'param_gen'])
TestParams = namedtuple('TestParams', ['in_shape', 'padding'])

TF_OPERATOR_MAP = {
    'maxwithnan': 'MAX',
    'max': 'MAX',
    'avg': 'AVG',
}


def get_grad_results(max_val, pool_op, input_shape, window_shape, stride_shape,
                     padding):
    """
    Compute pooling backprop values.

    Will create an input tensor of the required size filled with values 1, 2,
    3... and use these to compute the pooling, then create another tensor with
    the same values to use as the errors to backpropagate.
    Returns the computed values in a numpy array.
    """
    input = helpers.get_variable(input_shape, max_val)
    with tf.GradientTape() as tape:
        output = tf.nn.pool(input,
                            window_shape=window_shape,
                            pooling_type=TF_OPERATOR_MAP[pool_op],
                            strides=stride_shape,
                            padding=padding,
                            data_format="NHWC")
    output_shape = output.shape
    error = helpers.get_variable(output_shape, max_val)
    return tape.gradient(output, input, error)


def get_pool_results(max_val, pool_op, input_shape, window_shape, stride_shape,
                     padding):
    """
    Compute forward pooling.

    Will create an input tensor of the required size filled with values 1, 2,
    3... and use these to compute the pooling.
    Returns the computed values in a numpy array.
    """
    total_inp_size = np.product(input_shape)

    input_vals = helpers.get_tensor_data(total_inp_size, max_val)

    inp_tensor = tf.constant(input_vals,
                             shape=input_shape,
                             dtype=np.float64)
    return tf.nn.pool(inp_tensor,
                      window_shape=window_shape,
                      pooling_type=TF_OPERATOR_MAP[pool_op],
                      strides=stride_shape,
                      padding=padding,
                      data_format="NHWC")


def get_result_function(test_case):
    """
    Get the function which will compute the expected values for the given test case.
    """
    if (test_case.direction == 'grad'):
        return get_grad_results
    else:
        return get_pool_results


def get_result_and_size(test_case, test_params):
    """
    Get the result of the specified convolution and max input value.

    Ensures that the resulting values are less than the REQUIRED_MAX, and if
    not will adjust the maximum value to allow in the input tensors.
    """
    window_shape = [test_case.window, test_case.window]
    stride_shape = [test_case.stride, test_case.stride]
    return helpers.get_result_and_size(get_result_function(test_case),
                                       pool_op=test_case.test_type,
                                       input_shape=test_params.in_shape,
                                       window_shape=window_shape,
                                       stride_shape=stride_shape,
                                       padding=test_params.padding)


TEST_CASE_TPL = "{test_type}Window{window}Stride{stride}{direction}"
TEST_NAME_TPL = "{padding}{in_s[0]}x{in_s[1]}x{in_s[2]}x{in_s[3]}"
IN_SHAPE_INIT_TPL = "{{{{ {0[0]}, {0[1]}, {0[2]}, {0[3]} }}}}"

OPERATOR_MAP = {
    'maxwithnan': 'pooling::MaxWithNan',
    'max': 'pooling::Max',
    'avg': 'pooling::Average',
}

DIRECTION_MAP = {
    'forward': 'pooling::Forward',
    'grad': 'pooling::Backpropagate',
}


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
                                          stride=test_case.stride,
                                          direction=helpers.to_camel_case(
                                              test_case.direction))
    test_name = TEST_NAME_TPL.format(padding=test_params.padding,
                                     in_s=test_params.in_shape)
    in_shape_init = IN_SHAPE_INIT_TPL.format(test_params.in_shape)
    test_lines = [
        "TYPED_TEST({}, {}) {{".format(test_case_name, test_name),
        "  using DataType = typename TestFixture::DataType;",
        "  const std::vector<DataType> exp_out = {};".format(
            helpers.format_tensor(output)),
        "  const std::array<int, 4> in_shape = {};".format(in_shape_init),
        "  const auto padding = PaddingMode::{};".format(test_params.padding),
        "  const auto params = getPoolingParams<{}, {}>(in_shape, padding);".
        format(test_case.window, test_case.stride),
        "  const DataType max_input_val = {:.1f};".format(max_input_val),
        "  this->test_pool(exp_out, params, max_input_val);",
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
        for padding in PADDING_VALUES:
            yield TestParams(in_shape=in_shape, padding=padding)


def output_for_test_case(test_case):
    """
    Create a list of strings corresponding to separate lines in the full test
    case. The output contains headers, includes, setup and all the tests for
    the test case.
    """
    camel_case_type = helpers.to_camel_case(test_case.test_type)
    test_case_name = TEST_CASE_TPL.format(test_type=camel_case_type,
                                          window=test_case.window,
                                          stride=test_case.stride,
                                          direction=helpers.to_camel_case(
                                              test_case.direction))
    output = [
        TYPED_TEST_SUITE_DECL_TPL.format(
            test_case=test_case_name,
            operation=OPERATOR_MAP[test_case.test_type],
            direction=DIRECTION_MAP[test_case.direction])
    ]

    for test_params in test_case.param_gen(test_case):
        output.extend(get_test_lines(test_case, test_params))
    output.append("\n")
    return output


def get_initial_boilerplate():
    """ Get the boilerplate for the top of the test file. """
    scriptname = os.path.basename(__file__)
    return [
        helpers.get_license(),
        helpers.get_dont_modify_comment(scriptname=scriptname),
        INCLUDES,
        DATA_TYPES,
    ]


FILENAME_TPL = "pooling/{test_type}_window{window}_stride{stride}_{direction}.cc"


def get_test_case_filename(test_case):
    "Get filename for test case."
    return FILENAME_TPL.format(test_type=test_case.test_type,
                               window=test_case.window,
                               stride=test_case.stride,
                               direction=test_case.direction)


def test_cases(param_gen):
    "Test case generator giving all possible test cases."
    for window, stride in zip(WINDOW_LIST, STRIDE_LIST):
        for test_type, direction in itertools.product(TEST_TYPES, DIRECTIONS):
            yield TestCaseParams(test_type=test_type,
                                 window=window,
                                 stride=stride,
                                 direction=direction,
                                 param_gen=param_gen)


def generate_pooling_tests():
    np.set_printoptions(suppress=True, threshold=1000000, linewidth=1000000)
    test_dir = helpers.get_test_directory()
    os.chdir(test_dir)
    for test_case in test_cases(test_params_for_test_case):
        filename = get_test_case_filename(test_case)
        output = get_initial_boilerplate()
        output.extend(output_for_test_case(test_case))
        with open(filename, 'w') as f:
            f.write('\n'.join(output))
        print("File '{}' written".format(filename))


def small_fastdiv_test_params(test_case):
    """ Parameter generator to force the use of fast div kernels. """
    batches = [1]
    in_sizes = [test_case.window + test_case.stride + 2]
    channels = [5, 6, 8]
    for in_shape in itertools.product(batches, in_sizes, in_sizes, channels):
        for padding in PADDING_VALUES:
            yield TestParams(in_shape=in_shape, padding=padding)


def generate_fastdiv_tests():
    np.set_printoptions(suppress=True, threshold=1000000, linewidth=1000000)
    test_dir = helpers.get_test_directory()
    os.chdir(test_dir)
    filename = 'pooling/pooling_fastdiv.cc'
    output = get_initial_boilerplate()
    for test_case in test_cases(small_fastdiv_test_params):
        output.extend(output_for_test_case(test_case))
    with open(filename, 'w') as f:
        f.write('\n'.join(output))
    print("File '{}' written".format(filename))


if __name__ == "__main__":
    generate_pooling_tests()
    generate_fastdiv_tests()
