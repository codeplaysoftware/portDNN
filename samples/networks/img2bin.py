#! /usr/bin/env python3

import numpy as npy
from PIL import Image
from sys import argv


def main():
    average = npy.array([103.939, 116.779, 123.68], dtype=npy.float32)
    f = argv[1]
    img = Image.open(f).resize((224, 224))
    raw = npy.frombuffer(img.tobytes(), dtype=npy.uint8).astype(npy.float32)
    arr = raw.reshape(224 * 224, 3)
    # Convert RGB image to BGR
    arr = arr[..., ::-1]
    arr = arr - average
    arr.tofile(f + ".bin")


if __name__ == "__main__":
    main()
