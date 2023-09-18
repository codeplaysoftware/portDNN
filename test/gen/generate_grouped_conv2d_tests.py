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
TEST_SAMPLE_SIZE = 5
WINDOW_LIST = [1, 1, 3, 3, 5, 5, 7, 7, 11, 11]
STRIDE_LIST = [1, 2, 1, 2, 1, 2, 1, 4, 1, 4]
BATCHES = [1, 3]
CHANNELS = [1, 2, 4]
FEATURES = [1, 2, 4]
GROUPS = [2, 2, 3, 2, 3, 4, 4, 5, 6, 7]
PADDING_VALUES = ["SAME", "VALID"]
GROUP_FORMATS = ["STRIDED", "INTERLEAVED"]
FILTER_FORMATS = ["HWCF", "FHWC"]
TEST_TYPES = ["forward"]
INCLUDES = r"""
#include <gtest/gtest.h>

#include "portdnn/padding_mode.h"

#include "test/types/cartesian_product.h"
#include "test/types/data_format_types.h"
#include "test/types/kernel_data_types.h"
#include "test/types/nested_pairs_to_triple.h"
#include "test/types/test_backend_types.h"
#include "test/types/to_gtest_types.h"

#include "test/conv2d/selector_list.h"
#include "test/conv2d/group_convolution_fixture.h"

#include <array>
#include <vector>"""
DATA_TYPES = r"""
using DataTypeList = sycldnn::types::KernelDataTypes;
using Selectors = sycldnn::types::SelectorList;
using Backends = sycldnn::types::AllMatmulBackendTypes;

using SNNTypePairs =
    sycldnn::types::CartesianProduct<Selectors, DataTypeList>::type;
using BackendTypePairs =
    sycldnn::types::CartesianProduct<SNNTypePairs, Backends>::type;
using TestTriples =
    sycldnn::types::NestedPairsToTriple<BackendTypePairs>::type;

using GTestTypeTriples = sycldnn::types::ToGTestTypes<TestTriples>::type;
"""
TYPED_TEST_SUITE_DECL_TPL = r"""
template <typename Triple>
using {test_case} = GroupWindowStrideTest<Triple, {window}, {stride}, {groups}>;
TYPED_TEST_SUITE({test_case}, GTestTypeTriples);"""

TestCaseParams = namedtuple('TestCaseParams',
                            ['test_type', 'window', 'stride', 'groups'])
TestParams = namedtuple('TestParams', ['in_shape', 'features', 'padding',
                        'filter_format', 'group_format'])


def get_forward_grouped_conv_results(
        max_val,
        input_shape,
        filter_shape,
        stride_shape,
        padding,
        groups,
        group_format):
    """
    Compute forward convolution.

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
                             dtype=np.float32)
    fil_tensor = tf.constant(filter_vals,
                             shape=filter_shape,
                             dtype=np.float32)

    input_groups = tf.split(
        axis=3,
        num_or_size_splits=groups,
        value=inp_tensor)
    weights_groups = tf.split(axis=3, num_or_size_splits=groups,
                              value=fil_tensor)

    def group_conv(inp, ker): return tf.nn.conv2d(
        inp, ker, strides=stride_shape, padding=padding,
        data_format="NHWC")

    conv_groups = []
    for inp, ker in zip(input_groups, weights_groups):
        conv_result = group_conv(inp, ker)
        if group_format == "INTERLEAVED":
            conv_result = tf.expand_dims(conv_result, -1)
        conv_groups.append(conv_result)

    res = tf.concat(axis=-1, values=conv_groups)
    return res


def get_conv_fn(test_type):
    """
    Get the function which computes the convolution corresponding to the test type.
    """
    if test_type == "forward":
        return get_forward_grouped_conv_results
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
        test_case.window, test_case.window,
        int(test_params.in_shape[-1] / test_case.groups),
        test_params.features
    ]
    stride_shape = [1, test_case.stride, test_case.stride, 1]
    return helpers.get_result_and_size(
        conv_fn,
        max_input_val=4,
        input_shape=test_params.in_shape,
        filter_shape=filter_shape,
        stride_shape=stride_shape,
        groups=test_case.groups,
        padding=test_params.padding,
        group_format=test_params.group_format)


TEST_CASE_TPL = "{test_type}Window{window}Stride{stride}Groups{groups}"
TEST_NAME_TPL = "{padding}{filter_format}{group_format}{in_s[0]}x{in_s[1]}"
TEST_NAME_TPL += "x{in_s[2]}x{in_s[3]}x{features}"
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
                                          stride=test_case.stride,
                                          groups=test_case.groups)
    test_name = TEST_NAME_TPL.format(padding=test_params.padding,
                                     in_s=test_params.in_shape,
                                     features=test_params.features,
                                     filter_format=test_params.filter_format,
                                     group_format=test_params.group_format)
    in_shape_init = IN_SHAPE_INIT_TPL.format(test_params.in_shape)
    test_lines = [
        "TYPED_TEST({}, {}) {{".format(test_case_name, test_name),
        "  using DataType = typename TestFixture::DataType;",
        "  const std::vector<DataType> exp_out = {};".format(
            helpers.format_tensor(output)),
        "  const std::array<int, 4> in_shape = {};".format(in_shape_init),
        "  const int features = {};".format(test_params.features),
        "  const auto padding = sycldnn::PaddingMode::{};".format(
            test_params.padding),
        "  const auto filter_format = sycldnn::FilterFormat::{};".format(
            test_params.filter_format),
        "  const auto group_format = sycldnn::BatchFormat::{};".format(
            test_params.group_format),
        "  const DataType max_input_val = {:.1f};".format(max_input_val),
        "  this->run_{}_test(exp_out, in_shape, features, padding,".format(
            test_case.test_type),
        "  filter_format, group_format, max_input_val);",
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

    # Set the random seed
    rng = np.random.default_rng(23456)

    # Calculate the total number of shape combinations
    total_shape_cases = len(BATCHES) * (len(in_sizes)**2)

    # Get the set of valid parameter combinations
    param_cases = []
    for param_case in itertools.product(
            PADDING_VALUES, FILTER_FORMATS, GROUP_FORMATS, CHANNELS, FEATURES):
        if param_case[1] == "FHWC" and param_case[2] == "INTERLEAVED":
            continue
        param_cases.append(param_case)
    total_param_cases = len(param_cases)

    # Randomly assign `TEST_SAMPLE_SIZE` amount of parameter cases to each shape case.
    # This ensures that the parameter space is covered as well as the shape space without
    # overloading the tests with a full grid search.
    assignments = np.concatenate([rng.choice(
        range(total_param_cases),
        TEST_SAMPLE_SIZE, replace=False)] * total_shape_cases)

    for i, shape in enumerate(
            itertools.product(
            BATCHES, in_sizes, in_sizes)):
        shape_list = list(shape)
        shape_list.append(1)
        for j in range(TEST_SAMPLE_SIZE):
            padding, filter_format, group_format, channels, features = param_cases[
                assignments[i * TEST_SAMPLE_SIZE + j]]
            shape_list[-1] = channels * test_case.groups
            yield TestParams(in_shape=tuple(shape_list),
                             features=features * test_case.groups,
                             padding=padding, filter_format=filter_format,
                             group_format=group_format)


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
                                          stride=test_case.stride,
                                          groups=test_case.groups)
    output = [
        helpers.get_license(),
        helpers.get_dont_modify_comment(scriptname=scriptname), INCLUDES,
        DATA_TYPES,
        TYPED_TEST_SUITE_DECL_TPL.format(test_case=test_case_name,
                                         window=test_case.window,
                                         stride=test_case.stride,
                                         groups=test_case.groups)
    ]
    for test_params in test_params_for_test_case(test_case):
        output.extend(get_test_lines(test_case, test_params))
    return output


FILENAME_TPL = "conv2d/{test_type}_window{window}_stride{stride}_groups{groups}.cc"


def get_test_case_filename(test_case):
    "Get filename for test case."
    return FILENAME_TPL.format(test_type=test_case.test_type,
                               window=test_case.window,
                               stride=test_case.stride,
                               groups=test_case.groups)


def test_cases():
    "Test case generator giving all possible test cases."
    for window, stride, groups in zip(WINDOW_LIST, STRIDE_LIST, GROUPS):
        for test_type in TEST_TYPES:
            yield TestCaseParams(test_type=test_type,
                                 window=window,
                                 stride=stride,
                                 groups=groups)


def generate_grouped_conv2d_tests():
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
    generate_grouped_conv2d_tests()
