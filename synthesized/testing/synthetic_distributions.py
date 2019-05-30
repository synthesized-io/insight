import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import bernoulli
from scipy.stats import ks_2samp
from scipy.stats import powerlaw
from typing import Tuple, Dict
from matplotlib.pyplot import Axes

from synthesized.core import BasicSynthesizer
from synthesized.core.values import Value
from synthesized.testing.evaluation import Evaluation
from synthesized.core.values import CategoricalValue, ContinuousValue


def create_bernoulli(probability: float, size: int) -> pd.DataFrame:
    """Draws `size` samples from a Bernoulli distribution with probability of 1 0 <= `probability` <= 1."""
    return pd.DataFrame({"x": np.random.random_sample(size=size) < probability})


def create_categorical(probabilities: list, size: int) -> pd.DataFrame:
    """Draws `size` samples from a categorical distribution with
     `len(probabilities)` categories and probability `probabilities[i]`
     for class `i`.
     """
    categories = list(map(str, range(len(probabilities))))
    x = np.random.choice(a=categories, size=size, p=probabilities)
    return pd.DataFrame({"x": x})


def create_1d_gaussian(mean: float, std: float, size: int) -> pd.DataFrame:
    """Draws `size` samples from a one-dimensional gaussian distribution with params N(mean, std)."""
    x = mean + std*np.random.randn(size=size)
    return pd.DataFrame({"x": x})


def create_gauss_ball(x_mean: float, x_std: float, y_mean: float, y_std: float, size: int) -> pd.DataFrame:
    """Creates a two-dimensional (axes: x,y) gauss distribution with params N([x_mean, y_mean], [x_std, y_std])"""
    mean = [x_mean, y_mean]
    cov = [[x_std, 0], [0, y_std]]
    x, y = np.random.multivariate_normal(mean, cov, size).T
    df = pd.DataFrame({'x': x, 'y': y})
    return df


def create_gauss_line(x_range: Tuple[float, float], intercept: float, slope: float, y_std: float, size: int) -> pd.DataFrame:
    """Creates a two-dimensional (axes: x,y) cloud of points around a line `y=x*slope + intercept`
     with standard deviation y_std along y axis and confined to [x_range[0], x_range[1]] along x axis"""
    x = np.random.uniform(low=x_range[0], high=x_range[1], size=size)
    y = intercept + x*slope + np.random.normal(loc=0, scale=y_std, size=size)
    df = pd.DataFrame({'x': x, 'y': y})
    return df


def create_power_law_distribution(shape: float, scale: float, size: int):
    """Creates a one-dimensional (axis: x) power-low distribution with pdf `shape * x**(shape-1)`"""
    return pd.DataFrame({'x': scale*powerlaw.rvs(shape, size=size)})


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
    sample = [j for i in range(n_classes) for j in [i]*2**(n_classes-i-1)]
    df = pd.DataFrame({'x': sample})
    df = df.sample(size, replace=True)
    return df


def _plot_data(data: pd.DataFrame, ax: Axes, value_types: Dict[str, Value]) -> None:
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
            #sns.jointplot(x="x", y="y", data=data, kind="kde", ax=ax)
            ax.hist2d(data['x'], data['y'], bins=100)
    else:
        assert False


def synthesize_and_plot(data: pd.DataFrame, name: str, evaluation: Evaluation, num_iterations: int=None) -> None:
    if num_iterations is None:
        num_iterations = evaluation.config['num_iterations']
    start = time.time()
    with BasicSynthesizer(data=data, **evaluation.config['params']) as synthesizer:
        # print('value types:')
        # for value in synthesizer.values:
        #     print(value.name, value)
        value_types = {value.name: type(value) for value in synthesizer.values}
        synthesizer.learn(data=data, num_iterations=num_iterations)
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