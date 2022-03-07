#! /usr/bin/env python3

import numpy as npy
import h5py as h5
from sys import argv


def dump_data(name):
    f = h5.File(name, 'r')
    count = 0
    for layer in f.keys():
        try:
            weights = f[layer][layer + '_W_1:0']
            biases = f[layer][layer + '_b_1:0']
        except KeyError:
            continue

        output_w = npy.ndarray(weights.shape, dtype=npy.float32)
        output_b = npy.ndarray(biases.shape, dtype=npy.float32)
        count += 1
        layer_name = f'layer_{count}'
        weights.read_direct(output_w)
        biases.read_direct(output_b)
        with open(layer_name + '-weights.bin', 'wb') as f_weights:
            output_w.tofile(f_weights)
        with open(layer_name + '-biases.bin', 'wb') as f_biases:
            output_b.tofile(f_biases)


if __name__ == "__main__":
    for name in argv[1:]:
        dump_data(name=name)
