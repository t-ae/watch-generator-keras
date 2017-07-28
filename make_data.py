#!/usr/bin/env python

import numpy as np
import glob, os
from skimage.io import imread
from skimage.transform import resize, rotate

files_dir = os.path.dirname(__file__)
out_path = os.path.join(files_dir, "./train_data.npy")
jpgs = glob.glob(os.path.join(files_dir, "./watch/*.jpg"))

# train data
images = []
angles = []
max_angle = 15
for jpg in jpgs:
    print(jpg, end='          \r')
    image = imread(jpg)
    image = 2.0*image / 255 - 1
    image = image[:, 2:-2, :]
    image = np.pad(image, ((1, 1), (0, 0), (0, 0)), mode="edge")
    image = resize(image, [128, 96], mode="edge")
    image = image[8:-8, :, :]

    images.append(image)

images = np.stack(images)

np.save(out_path, images)
print("Save:", out_path)
print("{0} samples".format(len(images)))
print(images.shape)
