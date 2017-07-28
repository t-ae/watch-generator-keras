#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import utils
import network
from custom_layers import PixelShuffler, InstanceNormalization
import plot_tool

def scale(vector, length):
    return vector * length / np.linalg.norm(vector)


def main():
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = os.path.join(os.path.dirname(__file__), "./models/lsgan.h5")

    if model_path is None:
        print("Model file not found")
        exit(-1)

    with tf.device("/cpu:0"):
        from keras.models import load_model

        print("load: {0}".format(model_path))
        gan = load_model(model_path, custom_objects={"PixelShuffler": PixelShuffler,
                                                     "InstanceNormalization": InstanceNormalization})
        rows = 3

        z1 = utils.create_z(rows, gan.input_shape[-1])
        length = np.linalg.norm(z1)
        z2 = scale(utils.create_z(rows, gan.input_shape[-1]), length)

        zs = np.hstack([scale(p * z1 + (1-p) * z2, length) for p in np.arange(0, 1.01, 0.1)])
        zs = zs.reshape([-1, network.Z_DIMENSION])
        cols = len(zs)//rows

        images, *_ = gan.predict(zs)
        images = np.clip((images + 1) / 2, 0, 1)

        plot_tool.plot_images(images, rows, cols)

if __name__ == '__main__':
    main()
