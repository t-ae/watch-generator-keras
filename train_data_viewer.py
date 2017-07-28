#!/usr/bin/env python

import os
import numpy as np
import plot_tool
import utils

files_dir = os.path.dirname(__file__)
npy_path = os.path.join(files_dir, "./train_data.npy")
images = (np.load(npy_path))

np.random.shuffle(images)

rows = 5
cols = 7

images = utils.transform(images[:rows * cols])
images += utils.noise(000, images.shape)

images = np.clip((images + 1) / 2, 0, 1)

plot_tool.plot_images(images, None, rows, cols)