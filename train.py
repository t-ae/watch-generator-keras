#!/usr/bin/env python

import os
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
import network
import utils
import plot_tool


def iterate(x_train, x_history, batch_size):
    for i in range(len(x_train) // batch_size):
        yield x_train[i*batch_size:(i+1)*batch_size], x_history[i*batch_size:(i+1)*batch_size]


def main():

    a, b, c = 0, 1, 1

    # setup model
    generator = network.create_generator()
    generator.summary()
    discriminator = network.create_discriminator()
    discriminator.summary()

    d_opt = Adam(lr=1e-4, beta_1=0.1)
    discriminator.compile(optimizer=d_opt,
                          loss=network.create_lsgan_d_loss(a, b),
                          metrics=["accuracy"])

    discriminator.trainable = False
    for layer in discriminator.layers:
        layer.trainable = False
    g_opt = Adam(lr=2e-4, beta_1=0.5)
    in_z = Input([network.Z_DIMENSION])
    gen = generator(in_z)
    out = discriminator(gen)
    gan = Model(in_z, [gen, out])
    gan.compile(optimizer=g_opt,
                loss=[None, network.create_lsgan_g_loss(c)],
                metrics=["accuracy"])

    npy_path = os.path.join(os.path.dirname(__file__), "./train_data.npy")
    x_train = np.load(npy_path)

    # train
    batch_size = 16

    plotter = plot_tool.Plotter(128, 100)

    x_history = np.zeros([len(x_train), 112, 96, 3])

    for epoch in range(9999999):
        np.random.shuffle(x_train)
        np.random.shuffle(x_history)
        for step, (b_image, b_history) in enumerate(iterate(x_train, x_history, batch_size)):

            loss_g, _, acc_g = gan.train_on_batch(utils.create_z(batch_size), [c] * batch_size)

            b_image = utils.transform(b_image)
            b_image += utils.noise(epoch, b_image.shape)
            if epoch > 10:
                generated = generator.predict(utils.create_z(batch_size))
                fakes = np.vstack([generated, b_history])
            else:
                generated = generator.predict(utils.create_z(batch_size))
                fakes = generated
            images = np.vstack([fakes, b_image])
            labels = [a]*len(fakes) + [b]*len(b_image)
            loss_d, acc_d = discriminator.train_on_batch(images, labels)

            plotter.append(loss_g, loss_d, acc_g, acc_d)

            print(f"{epoch:04d}-{step:03d}: g:{loss_g:.9f} d:{loss_d:.9f}")

        # save
        gan.save("./models/gan.h5", include_optimizer=False)

        # np.save("./history.npy", h_array)
        if epoch % 100 == 0:
            gan.save("./models/{}.h5".format(epoch), include_optimizer=False)

        generated = generator.predict(utils.create_z(100))
        x_history = np.vstack([x_history[100:], generated])

if __name__ == '__main__':
    import tensorflow as tf
    from keras import backend as K

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    main()
