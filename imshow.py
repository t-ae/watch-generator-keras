#!/usr/bin/env python

import sys
import numpy as np
import plot_tool


def main():
    npy_path = sys.argv[1]

    images = np.load(npy_path)

    rows = 4
    cols = 10

    images = np.clip((images + 1) / 2, 0, 1)

    plot_tool.plot_images(images, rows, cols)

if __name__ == '__main__':
    main()
