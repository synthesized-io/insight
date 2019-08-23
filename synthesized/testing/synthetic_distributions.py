import time
from typing import Dict, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import Axes
from numpy.random import binomial
from numpy.random import exponential, normal
from scipy.stats import bernoulli
from scipy.stats import ks_2samp
from scipy.stats import powerlaw

from . import Evaluation
from ..common.values import CategoricalValue, ContinuousValue
from ..highdim import HighDimSynthesizer


def product(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    for name, column in df1.items():
        df.loc[:, name + '1'] = column
    for name, column in df2.items():
        df.loc[:, name + '2'] = column
    return df


def create_bernoulli(probability: float, size: int) -> pd.DataFrame:
    """Draws `size` samples from a Bernoulli distribution with probability of 1 0 <= `probability` <= 1."""
    x = np.random.random_sample(size=size) < probability
    return pd.DataFrame({'x': x})


def create_categorical(probabilities: list, size: int) -> pd.DataFrame:
    """Draws `size` samples from a categorical distribution with
     `len(probabilities)` categories and probability `probabilities[i]`
     for class `i`.
     """
    categories = list(range(len(probabilities)))
    x = np.random.choice(a=categories, size=size, p=probabilities)
    return pd.DataFrame({'x': x})


def create_1d_gaussian(mean: float, std: float, size: int) -> pd.DataFrame:
    """Draws `size` samples from a one-dimensional gaussian distribution with params N(mean, std)."""
    x = np.random.normal(loc=mean, scale=std, size=size)
    return pd.DataFrame({'x': x})


def create_gauss_ball(x_mean: float, x_std: float, y_mean: float, y_std: float, size: int) -> pd.DataFrame:
    """Creates a two-dimensional (axes: x,y) gauss distribution with params N([x_mean, y_mean], [x_std, y_std])"""
    mean = [x_mean, y_mean]
    cov = [[x_std, 0], [0, y_std]]
    x, y = np.random.multivariate_normal(mean, cov, size).T
    df = pd.DataFrame({'x': x, 'y': y})
    return df


def create_two_1d_gaussians(
    mean1: float, std1: float, size1: int, mean2: float, std2: float, size2: int
) -> pd.DataFrame:
    """Creates a mixture of 1-dimensional distributions"""
    df1 = pd.DataFrame({"x": normal(loc=mean1, scale=std1, size=size1)})
    df2 = pd.DataFrame({"x": normal(loc=mean2, scale=std2, size=size2)})
    df = pd.concat([df1, df2])
    return df


def create_exp_gaussian_mixture(mean1: float, std1: float, size1: int, scale: float, size2: int) -> pd.DataFrame:
    """Creates a mixture of 1-dimensional distributions"""
    df1 = pd.DataFrame({"x": normal(loc=mean1, scale=std1, size=size1)})
    df2 = pd.DataFrame({"x": exponential(scale=scale, size=size2)})
    df = pd.concat([df1, df2])
    return df


def create_three_1d_gaussians(mean1: float, std1: float, size1: int,
                              mean2: float, std2: float, size2: int,
                              mean3: float, std3: float, size3: int) -> pd.DataFrame:
    """Creates a mixture of 1-dimensional distributions"""
    df1 = pd.DataFrame({"x": normal(loc=mean1, scale=std1, size=size1)})
    df2 = pd.DataFrame({"x": normal(loc=mean2, scale=std2, size=size2)})
    df3 = pd.DataFrame({"x": normal(loc=mean3, scale=std3, size=size3)})
    df = pd.concat([df1, df2, df3])
    return df


def create_two_gaussian_mixtures(mean1: float, std1: float, mean2: float, std2: float, size: int) -> pd.DataFrame:
    """Creates a mixture of 1-dimensional distributions (a proper one)"""
    size1 = binomial(n=size, p=0.5)
    size2 = size - size1
    df1 = pd.DataFrame({"x": normal(loc=mean1, scale=std1, size=size1)})
    df2 = pd.DataFrame({"x": normal(loc=mean2, scale=std2, size=size2)})
    df = pd.concat([df1, df2])
    df = df.sample(frac=1)
    return df


def create_gauss_line(
    x_range: Tuple[float, float], intercept: float, slope: float, y_std: float, size: int
) -> pd.DataFrame:
    """Creates a two-dimensional (axes: x,y) cloud of points around a line `y=x*slope + intercept`
     with standard deviation y_std along y axis and confined to [x_range[0], x_range[1]] along x axis"""
    x = np.random.uniform(low=x_range[0], high=x_range[1], size=size)
    y = intercept + x * slope + np.random.normal(loc=0, scale=y_std, size=size)
    df = pd.DataFrame({'x': x, 'y': y})
    return df


def create_power_law_distribution(shape: float, scale: float, size: int):
    """Creates a one-dimensional (axis: x) power-low distribution with pdf `shape * x**(shape-1)`"""
    return pd.DataFrame({'x': scale * powerlaw.rvs(shape, size=size)})


def create_bernoulli_distribution(ratio: float, size: int) -> pd.DataFrame:
    """Creates a one-dimensional (axis: x) bernoulli distribution with probability of 1-s equal to `ratio`"""
    df = pd.DataFrame({'x': bernoulli.rvs(ratio, size=size)})
    return df


def create_conditional_distibution(*norm_params: Tuple[float, float], size: int) -> pd.DataFrame:
    """Creates a two-dimensional (axes: x,y) distribution where y has values N_i(mean_i, std_i) and x=i where
     i=0..len(norm_params), norm_param is a sequence of (mean_i, std_i)"""
    df = pd.DataFrame()
    for i, (mean, std) in enumerate(norm_params):
        x = [i] * size
        y = np.random.normal(mean, std, size)
        df = df.append(pd.DataFrame({'x': x, 'y': y}), ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def create_unifom_categorical(n_classes: int, size: int) -> pd.DataFrame:
    """Creates a one-dimensional (axis: x) unif{0, n_classes-1} distribution"""
    df = pd.DataFrame({'x': range(n_classes)})
    df = df.sample(size, replace=True)
    return df


def create_power_law_categorical(n_classes: int, size: int) -> pd.DataFrame:
    """Creates a one-dimensional (axis: x) distribution where each class has 2 times less elements than previous one"""
    sample = [j for i in range(n_classes) for j in [i] * 2 ** (n_classes - i - 1)]
    df = pd.DataFrame({'x': sample})
    df = df.sample(size, replace=True)
    return df


def create_time_series_data(func, length):
    """
    Create a data frame of a time-series based on a time-series
    function.

    :param func: {times: np.array[int]} -> np.array[float]]
                a function that takes a sequence of time steps and noise
                and returns a series of values
    :param length: [int] number of time steps
    :return: pd.DataFrame{t[datetime], x[float]}
    """
    # create time columns
    times = np.arange(start=0, stop=length)

    # create value column
    xs = func(times)

    # cast times to date-time
    times = pd.to_datetime(times, unit="d")

    # combine
    df = pd.DataFrame({"t": times, "x": xs})
    return df


def additive_linear(a, b, sd):
    """
    A linear trend with additive noise
    :param a: [float] slope
    :param b: [float] intercept
    :param sd: [float] error standard deviation
    """

    def out_func(times):
        eps = sd * np.random.randn(times.shape[0])
        return a * times + b + eps

    return out_func


def additive_sine(a, p, sd):
    """
    A sinusoidal trend with additive noise
    :param a: [float] amplitude
    :param p: [float] period
    :param sd: [float] error standard deviation
    """

    def out_func(times):
        eps = sd * np.random.randn(times.shape[0])
        return a * np.sin(2 * np.pi * times / p) + eps

    return out_func


def auto_regressive(phi, c, sd):
    """
    A linear autoregressive process of order k
    i.e an AR(k) process
    :param phi: [np.array] regression weights
    :param c: [float] bias
    :param sd: [float] error standard deviation
    """
    k = phi.shape[0]

    def out_func(times):
        eps = sd * np.random.randn(times.shape[0])
        out_list = k * [0.]
        for i in range(times.shape[0]):
            # fetch regression context: previous k values
            x_prev = np.array(out_list[-k:][::-1])
            # sample next value
            x_t = c + (phi * x_prev).sum() + eps[i]
            out_list.append(x_t)
        return out_list[k:]

    return out_func


def add_series(*args):
    """
    Return a time series which is a sum of
    several other time series.
    :param args: times series closures
    :return: time series closure
    """

    def out_func(times):
        series = [func(times) for func in args]
        return sum(series)

    return out_func


def rolling_mse_asof(data, synthesized, sd, time_unit=None):
    """
    Calculate the mean-squared error between the "x" values of the original and synthetic
    data. The sets of times may not be identical so we use "as of" (last observation rolled
    forward) to interpolate between the times in the two datasets.

    The dates are also optionally truncated to some unit following the syntax for the pandas
    `.floor` function.

    :param data: [pd.DataFrame{id[int], t[datetime], x[float]}] original data
    :param synthesized: [pd.DataFrame{id[int], t[datetime], x[float]}] synthesized data
    :param sd: [float] error standard deviation
    :param time_unit: [str] the time unit to round to. See documentation for pandas `.floor` method.
    :return: [(float, float)] MSE and MSE/(2*error variance)
    """
    # truncate date
    if time_unit is not None:
        synthesized.t = synthesized.t.dt.floor(time_unit)
        data.t = data.t.dt.floor(time_unit)

    # join datasets
    joined = pd.merge_asof(data[["t", "x"]], synthesized[["t", "x"]], on="t")

    # calculate metrics
    mse = ((joined.x_x - joined.x_y) ** 2).mean()
    mse_eff = mse / (2 * sd ** 2)

    return mse, mse_eff


def _plot_data(data: pd.DataFrame, ax: Axes, value_types: Dict[str, Type]) -> None:
    if len(value_types) == 1:
        if value_types['x'] is CategoricalValue:
            return sns.distplot(data, ax=ax, kde=False)
        else:
            return sns.distplot(data, ax=ax)
    elif len(value_types) == 2:
        assert value_types['y'] is ContinuousValue
        if value_types['x'] is CategoricalValue:
            sns.violinplot(x="x", y="y", data=data, ax=ax)
        else:
            # sns.jointplot(x="x", y="y", data=data, kind="kde", ax=ax)
            ax.hist2d(data['x'], data['y'], bins=100)
    else:
        assert False


def synthesize_and_plot(data: pd.DataFrame, name: str, evaluation: Evaluation, num_iterations: int = None) -> None:
    if num_iterations is None:
        num_iterations = evaluation.config['num_iterations']
    start = time.time()
    with HighDimSynthesizer(df=data, **evaluation.config['params']) as synthesizer:
        # print('value types:')
        # for value in synthesizer.values:
        #     print(value.name, value)
        value_types = {value.name: type(value) for value in synthesizer.values}
        synthesizer.learn(df_train=data, num_iterations=num_iterations)
        print()
        print('took', time.time() - start, 's')
        synthesized = synthesizer.synthesize(n=len(data))
        distances = [ks_2samp(data[col], synthesized[col])[0] for col in data.columns]
        avg_distance = np.mean(distances)
        evaluation[name + '_avg_distance'] = avg_distance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
        ax1.set_title('original')
        ax2.set_title('synthesized')
        _plot_data(data, ax=ax1, value_types=value_types)
        _plot_data(synthesized, ax=ax2, value_types=value_types)
