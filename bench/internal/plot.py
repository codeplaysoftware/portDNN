#!/usr/bin/python3
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

from __future__ import print_function

try:
    # With python3 `zip` returns an iterator, however with python2, use
    # `itertools.izip` instead
    import itertools.izip as zip
except ImportError:
    pass

import argparse
import re

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_benchmark(filename):
    """ Load benchmark data from file. """
    header_line = 0
    with open(filename) as inp:
        for line in inp:
            if line[0:4] == 'name':
                break
            header_line += 1
    data = pd.read_csv(filename, header=header_line)
    return data


def clean_benchmark_data(x):
    """ Clean the benchmark data in DataFrame x. """
    y = x.drop(
        [
            'name', 'time_unit', 'bytes_per_second', 'error_occurred',
            'error_message', 'batch', 'bytes_read', 'bytes_written',
            'channels', 'features', 'in_cols', 'in_rows', 'out_cols',
            'out_rows', 'pad_cols', 'pad_rows'
        ],
        axis=1)

    benchmark_method = x.name.str.split('/', expand=True)[1]
    y['conv_type'] = benchmark_method.str.split('_', expand=True)[0]
    y['tuple'] = (
        "(" + y.tile_rows.astype(str) + "," + y.tile_cols.astype(str) + "," +
        y.ch_vect.astype(str) + "," + y.feat_vect.astype(str) + ")")
    y['vectors'] = (
        "(" + y.ch_vect.astype(str) + "," + y.feat_vect.astype(str) + ")")
    return y


def get_device_and_driver(data):
    """ Get the device and driver version from the benchmark label. """
    label = data.label.iloc[0]
    search = re.search(r'device_name=(.*?),', label)
    device = search.group(1)
    search = re.search(r'driver_version=(.*?),', label)
    driver = search.group(1)
    return device, driver


def get_conv_type(data):
    """ Get the convolution type in data. """
    conv_type = data.conv_type.iloc[0]
    assert (data.conv_type == conv_type).all()
    return conv_type


def get_filter_sizes(data):
    """ Get the row and column size of the convolution filter in data. """
    fil_row = data.fil_rows.iloc[0]
    fil_col = data.fil_cols.iloc[0]
    assert (data.fil_rows == fil_row).all()
    assert (data.fil_cols == fil_col).all()
    return fil_row, fil_col


def get_stride_sizes(data):
    """ Get the row and column stride sizes of the convolution in data. """
    stride_row = data.stride_rows.iloc[0]
    stride_col = data.stride_cols.iloc[0]
    assert (data.stride_rows == stride_row).all()
    assert (data.stride_cols == stride_col).all()
    return stride_row, stride_col


def get_conv_title(data, main_title):
    """ Get a title based on main_title with information from the data. """
    device, driver = get_device_and_driver(data)
    conv_type = get_conv_type(data)
    fil_rows, fil_cols = get_filter_sizes(data)
    stride_rows, stride_cols = get_stride_sizes(data)
    conv_info = 'for a {}x{} {} convolution'.format(fil_rows, fil_cols,
                                                    conv_type)
    stride_info = 'with {}x{} strides'.format(stride_rows, stride_cols)
    device_info = 'Device = {}, Driver = {}'.format(device, driver)
    return '\n'.join(
        [' '.join([main_title, conv_info, stride_info]), device_info])


def get_readable_float_fmt_string(min_val, max_val):
    """
    Get a string format string to convert large floats into a human readable
    form, and the divisor required for the data.

    This takes both a max and min value to use to compute the best divisor.
    Currently only min_val is used, however this could be changed in the future.
    """
    exp_string = {
        1: '',
        1e3: 'Kilo',
        1e6: 'Mega',
        1e9: 'Giga',
        1e12: 'Tera',
        1e15: 'Peta'
    }
    exp_list = sorted(exp_string.keys())

    for low_exp, high_exp in zip(exp_list, exp_list[1:]):
        if (min_val < high_exp):
            return low_exp, '.1f', exp_string[low_exp]
    return 1, '.3e', ''


def plot_against_tuple(data):
    """ Plot a bar graph of flops against tuple comparing fast_div performance. """

    def _get_title(data):
        main_title = 'Flops with and without fast_div'
        return get_conv_title(data, main_title)

    if (len(data.index) == 0):
        print("Skipping plot, as dataframe is empty")
        return

    fg = sns.catplot(
        x='tuple', y='items_per_second', data=data, hue='fast_div', kind='bar')
    fg.set_axis_labels('(tile_rows, tile_cols, ch_vector, feat_vector)',
                       'Flops')
    fg.ax.set_xticklabels(fg.ax.get_xticklabels(), rotation=90)
    fg.fig.subplots_adjust(top=0.9)
    fg.fig.suptitle(_get_title(data))
    return fg


def plot_grid(data):
    """ Plot a grid of heatmaps showing performance for each tile pairing. """

    def _draw_heatmap(*args, **kwargs):
        """ Draw a heatmap showing flops for (ch_vect x feat_vect). """
        data = kwargs.pop('data')
        reshaped = pd.pivot_table(
            data,
            index='ch_vect',
            columns='feat_vect',
            values='items_per_second')
        reshaped = reshaped.sort_index(ascending=False, axis=0)
        sns.heatmap(reshaped, **kwargs)

    def _get_title(prefix, data):
        main_title = '{}Flops for different tile sizes and vector sizes'.format(
            prefix)
        return get_conv_title(data, main_title)

    if (len(data.index) == 0):
        print("Skipping plot, as dataframe is empty")
        return

    vmin = data['items_per_second'].min()
    vmax = data['items_per_second'].max()
    div, fmt, prefix = get_readable_float_fmt_string(vmin, vmax)
    tile_row_order = sorted(data['tile_rows'].unique(), reverse=True)
    tile_col_order = data['tile_cols'].unique().sort()
    scaled_data = data.copy()
    scaled_data['items_per_second'] = scaled_data['items_per_second'] / div
    scaled_vmin = vmin / div
    scaled_vmax = vmax / div
    fg = sns.FacetGrid(
        scaled_data,
        row='tile_rows',
        col='tile_cols',
        row_order=tile_row_order,
        col_order=tile_col_order,
        margin_titles=True)
    fg.map_dataframe(
        _draw_heatmap,
        annot=True,
        fmt=fmt,
        cmap='YlGnBu',
        vmin=scaled_vmin,
        vmax=scaled_vmax,
        cbar=False)
    fg.set_titles(
        template='Tile size: {row_name}x{col_name}',
        row_template='{row_name} rows per tile',
        col_template='{col_name} cols per tile')
    fg.set_axis_labels('Feature vectors', 'Channel vectors')
    fg.fig.subplots_adjust(top=0.9, hspace=0.25, wspace=0.15)
    fg.fig.suptitle(_get_title(prefix, data))
    return fg


def main():
    parser = argparse.ArgumentParser(
        description='Plot heatmaps of benchmark results.')
    parser.add_argument("file", help='Filename of benchmark csv')
    args = parser.parse_args()

    x = load_benchmark(args.file)
    x = clean_benchmark_data(x)
    plot_grid(x[(x.fast_div == 0) & (x.conv_type == 'Forward')])
    plot_grid(x[(x.fast_div == 0) & (x.conv_type == 'InputBackprop')])
    plot_against_tuple(x[x.conv_type == 'Forward'])
    plot_against_tuple(x[x.conv_type == 'InputBackprop'])
    plt.show()


if __name__ == "__main__":
    main()
