#! /usr/bin/env python3

import numpy as npy
import h5py as h5
from sys import argv


def dump_data(name):
    f = h5.File(name, 'r')
    for layer in f.keys():
        try:
            for k in f[layer][layer].keys():
                mystr = layer + '_' + k[:-2] + '.bin'
                print(mystr)
                weights = f[layer][layer][k]
                output_w = npy.ndarray(weights.shape, dtype=npy.float32)
                weights.read_direct(output_w)
                with open(mystr, 'wb') as f_weights:
                    output_w.tofile(f_weights)
        except KeyError:
            continue


if __name__ == "__main__":
    for name in argv[1:]:
        dump_data(name=name)
