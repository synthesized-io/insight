import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from synthesized.testing.test_data import generate_uniform, add_relative_noise

matplotlib.use('TkAgg')

np.random.seed(0)

if __name__ == '__main__':
    data = generate_uniform((20, 2)) * 100
    print(data)
    data_with_noise = add_relative_noise(data, 0.01)
    print(data_with_noise)
    plt.scatter(data[:, 0], data[:, 1])
    plt.scatter(data_with_noise[:, 0], data_with_noise[:, 1])
    plt.show()
