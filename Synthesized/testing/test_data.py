import numpy as np


def generate_uniform(shape):
    return np.random.rand(*shape)


def add_norm_noise(a, std):
    noise = np.random.normal(loc=0.0, scale=std, size=a.shape)
    return a + noise


def add_relative_noise(a, std):
    norms = np.abs(a)
    noise = np.random.normal(loc=0.0, scale=std, size=a.shape)
    return a + np.multiply(noise, norms)

