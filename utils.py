
import numpy as np
import network


def create_z(size, length=network.Z_DIMENSION):
    z = np.random.normal(0, 0.5, [size, length])
    return z


def transform(images, move=(4, 2)):
    moved = []
    for image in images:
        padded = np.pad(image, ((move[0], move[0]+1), (move[1], move[1] + 1), (0, 0)), "edge")
        x = np.random.randint(0, 2*move[1] + 1)
        y = np.random.randint(0, 2*move[0] + 1)
        moved.append(padded[y:-(2*move[0]+1) + y, x:-(2*move[1]+1) + x])
    return np.stack(moved)


def noise(epoch, shape):
    sigma = 0.3 / 2**(epoch/3000)
    return np.random.normal(0, sigma, shape)
