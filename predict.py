#!/usr/bin/env python

import os, sys
import tensorflow as tf
import numpy as np
import utils
from custom_layers import PixelShuffler, InstanceNormalization
import plot_tool

np.random.seed(42)


def main():
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = os.path.join(os.path.dirname(__file__), "./models/gan.h5")

    if model_path is None:
        print("Model file not found")
        exit(-1)

    with tf.device("/cpu:0"):
        from keras.models import load_model

        print("load: {0}".format(model_path))
        gan = load_model(model_path, custom_objects={"PixelShuffler": PixelShuffler, "InstanceNormalization": InstanceNormalization})

        rows = 3
        cols = 5

        z = utils.create_z(rows * cols, gan.input_shape[-1])
        images, score = gan.predict(z)
        images = np.clip((images + 1) / 2, 0, 1)

        plot_tool.plot_images(images, score, rows, cols)

if __name__ == '__main__':
    main()
