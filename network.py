#!/usr/bin/env python

from keras.models import Sequential, Model
from keras.layers import Input, InputLayer, Dense, Flatten, Reshape, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import AvgPool2D
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.losses import mean_squared_error
import keras.backend as K

from custom_layers import InstanceNormalization

Z_DIMENSION = 128


def create_lsgan_d_loss(a, b):
    def loss_func(y_true, y_pred):
        a_mask = K.cast(K.equal(y_true, a), K.floatx())
        b_mask = K.cast(K.equal(y_true, b), K.floatx())
        a_loss = K.sum((y_pred * a_mask - a) ** 2) / K.sum(a_mask)
        b_loss = K.sum((y_pred * b_mask - b) ** 2) / K.sum(b_mask)
        return (a_loss + b_loss) / 2
    return loss_func


def create_lsgan_g_loss(c):
    def loss_func(y_true, y_pred):
        return mean_squared_error(y_true, y_pred) / 2
    return loss_func


def create_discriminator(out="linear"):

    normalization = InstanceNormalization

    return Sequential([
        InputLayer([112, 96, 3]),
        Conv2D(32, 7, padding="same", use_bias=False),
        normalization(),
        ELU(),
        AvgPool2D(),
        Conv2D(64, 5, padding="same", use_bias=False),
        normalization(),
        ELU(),
        AvgPool2D(),
        Conv2D(128, 3, padding="same", use_bias=False),
        normalization(),
        ELU(),
        AvgPool2D(),
        Conv2D(256, 3, padding="same", use_bias=False),
        normalization(),
        ELU(),
        AvgPool2D(),
        Flatten(),
        Dense(1),
        Activation(out)
    ])


def create_generator():

    normalization = BatchNormalization

    inp = Input([Z_DIMENSION])
    x = Dense(7*6*256, use_bias=False)(inp)
    x = normalization()(x)
    x = ELU()(x)
    x = Reshape([7, 6, 256])(x)
    x = Conv2DTranspose(256, 3, padding="same", strides=2, use_bias=False)(x)  # 14x12
    x = normalization()(x)
    x = ELU()(x)
    x = Conv2DTranspose(256, 3, padding="same", strides=2, use_bias=False)(x)  # 28x24
    x = normalization()(x)
    x = ELU()(x)
    x = Conv2DTranspose(128, 3, padding="same", strides=2, use_bias=False)(x)  # 56x48
    x = normalization()(x)
    x = ELU()(x)
    x = Conv2DTranspose(64, 3, padding="same", strides=2, use_bias=False)(x)  # 112x96
    x = normalization()(x)
    x = ELU()(x)
    x = Conv2DTranspose(3, 5, padding="same", strides=1)(x)
    x = Activation("tanh")(x)
    return Model(inp, x, name="generator")
