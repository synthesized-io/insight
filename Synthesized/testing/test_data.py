import numpy as np
from numpy import linalg as LA


def generate_uniform(shape):
    return np.random.rand(*shape)


def add_norm_noise(a, std):
    noise = np.random.normal(loc=0.0, scale=std, size=a.shape)
    return a + noise


def add_relative_noise(a, std):
    norms = LA.norm(x=a, ord=None, axis=1)
    noise = np.random.normal(loc=0.0, scale=std, size=a.shape)
    return a + (noise.T * norms).T


import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = generate_uniform((20, 2))*100
    print(data)
    data_with_noise = add_relative_noise(data, 0.01)
    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(data_with_noise[:, 0], data_with_noise[:, 1])
    plt.show()
