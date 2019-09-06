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


def create_line(x_range: tuple, intercept: float, slope: float, y_std: float, size: int):
    """
    Draw `size` samples from a joint distribution where the marginal distribution of x
    is Uniform(`x_range[0]`, `x_range[1]`), and the conditional distribution of y given x
    is N(slope*x + intercept, y_std**2).
     """
    x = np.random.uniform(low=x_range[0], high=x_range[1], size=size)
    y = intercept + x*slope + np.random.normal(loc=0, scale=y_std, size=size)
    df = pd.DataFrame({'x': x, 'y': y})
    return df


def create_bernoulli(probability: float, size: int) -> pd.DataFrame:
    """Draws `size` samples from a Bernoulli distribution with probability of 1 0 <= `probability` <= 1."""
    df = pd.DataFrame({'x': bernoulli.rvs(probability, size=size).astype(str)})
    return df


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


def create_gauss_ball(x_mean: float,
                      x_std: float,
                      y_mean: float,
                      y_std: float,
                      size: int,
                      cor: float = 0.) -> pd.DataFrame:
    """Creates a two-dimensional (axes: x,y) gauss distribution with params N([x_mean, y_mean], [x_std, y_std])"""
    mean = [x_mean, y_mean]
    cov = [[x_std ** 2, x_std * y_std * cor], [x_std * y_std * cor, y_std ** 2]]
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
    df = pd.DataFrame({'x': bernoulli.rvs(ratio, size=size).astype(str)})
    return df


def create_conditional_distribution(*norm_params: Tuple[float, float], size: int) -> pd.DataFrame:
    """Creates a two-dimensional (axes: x,y) distribution where y has values N_i(mean_i, std_i) and x=i where
     i=0..len(norm_params), norm_param is a sequence of (mean_i, std_i)"""
    df = pd.DataFrame()
    for i, (mean, std) in enumerate(norm_params):
        x = [str(i)] * size
        y = np.random.normal(mean, std, size)
        df = df.append(pd.DataFrame({'x': x, 'y': y}), ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def create_uniform_categorical(n_classes: int, size: int) -> pd.DataFrame:
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


def create_mixed_continuous_categorical(n_classes, prior_mean=0, prior_sd=5, sd=1, size=10000):
    """
    Draw `size` samples from a joint distribution with one categorical and one continuous random variable.

    The categorical variable is drawn from a uniform distribution and each of whose `n_classes` values is
    associated with a mean drawn from a N(prior_mean, prior_sd) distribution. The continuous variable
    is drawn from a N(mean[cat], sd) distribution where `mean[cat]` is the mean for the value of the
    categorical variable.
    """
    means = prior_mean + prior_sd*np.random.randn(n_classes)
    categories = np.array([f"v_{i}" for i in list(range(n_classes))])
    discrete = np.random.choice(list(range(n_classes)), size=size)
    zs = np.random.randn(size)
    sample_means = means[discrete]
    values = categories[discrete]
    continuous = sample_means + sd*zs
    return pd.DataFrame({"x": values, "y": continuous})


def create_multidimensional_categorical(dimensions: int, n_classes: int, size: int, prefix: str = "x") -> pd.DataFrame:
    """
    Draw `size` samples of `dimensions` uniform, independent categorical random variables taking one of
    `n_classes` values.
    """
    categories = np.array([f"v_{i}" for i in list(range(n_classes))])
    z = np.random.choice(categories, dimensions*size).reshape(size, dimensions)
    columns = ["{}_{}".format(prefix, i) for i in range(dimensions)]
    df = pd.DataFrame(z, columns=columns)
    return df


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
        num_iterations = evaluation.configs['num_iterations']
    start = time.time()
    with HighDimSynthesizer(df=data, **evaluation.configs['params']) as synthesizer:
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
        evaluation.record_metric(evaluation=name, key='avg_distance', value=avg_distance)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
        ax1.set_title('original')
        ax2.set_title('synthesized')
        _plot_data(data, ax=ax1, value_types=value_types)
        _plot_data(synthesized, ax=ax2, value_types=value_types)
