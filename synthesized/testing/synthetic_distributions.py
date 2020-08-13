from random import choice, shuffle, gauss
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.random import binomial
from numpy.random import exponential, normal
from scipy.stats import bernoulli
from scipy.stats import powerlaw


def product(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    for name, column in df1.items():
        df.loc[:, name + '1'] = column
    for name, column in df2.items():
        df.loc[:, name + '2'] = column
    return df


def create_line(x_range: tuple, intercept: float, slope: float, y_std: float, size: int) -> pd.DataFrame:
    """
    Draw `size` samples from a joint distribution where the marginal distribution of x
    is Uniform(`x_range[0]`, `x_range[1]`), and the conditional distribution of y given x
    is N(slope * x + intercept, y_std**2).
     """
    x = np.random.uniform(low=x_range[0], high=x_range[1], size=size)
    y = intercept + x * slope + np.random.normal(loc=0, scale=y_std, size=size)
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


def create_gauss_ball(x_mean: float, x_std: float, y_mean: float, y_std: float,
                      size: int, cor: float = 0.) -> pd.DataFrame:
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
    """Creates a two-dimensional (axes: x,y) cloud of points around a line `y=x * slope + intercept`
     with standard deviation y_std along y axis and confined to [x_range[0], x_range[1]] along x axis"""
    x = np.random.uniform(low=x_range[0], high=x_range[1], size=size)
    y = intercept + x * slope + np.random.normal(loc=0, scale=y_std, size=size)
    df = pd.DataFrame({'x': x, 'y': y})
    return df


def create_power_law_distribution(shape: float, scale: float, size: int) -> pd.DataFrame:
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
        x = [i] * size
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


def create_mixed_continuous_categorical(n_classes: int, prior_mean: float = 0, prior_sd: float = 5,
                                        sd: float = 1, size: int = 10000) -> pd.DataFrame:
    """
    Create a dataset with one column drawn from a uniform categorical distribution over
    `n_classes` and one column drawn from a normal distribution whose parameters depend on
    the value of the categorical column.

    For each of the `n_classes` possible values of the categorical column, a mean is sampled from
    `N(prior_mean, prior_sd)`. The values of the continuous column are sampled from `N(mean[c], sd)`
    where `mean[c]` is the mean for the value of the categorical column.
    """
    means = prior_mean + prior_sd * np.random.randn(n_classes)
    categories = np.array([f"v_{i}" for i in list(range(n_classes))])
    discrete = np.random.choice(list(range(n_classes)), size=size)
    zs = np.random.randn(size)
    sample_means = means[discrete]
    values = categories[discrete]
    continuous = sample_means + sd * zs
    return pd.DataFrame({"x": values, "y": continuous})


def create_correlated_categorical(n_classes: int, size: int, sd: float = 1.) -> pd.DataFrame:
    """
    Create a dataset with a pair of dependent categorical columns each taking one of `n_classes`
    possible values.

    The first column is uniform distributed and values of the second are drawn conditional on the first.
    To specify the conditional distributions, for each of the `n_classes` possible values of the first column,
    a vector of logits is drawn from `N(0, sd * I_{n_classes, n_classes})`.
    """
    logits = sd * np.random.randn(n_classes * n_classes).reshape(n_classes, n_classes)
    categories = np.array([f"v_{i}" for i in list(range(n_classes))])
    independent = np.random.choice(list(range(n_classes)), size=size)
    gumbels = -np.log(-np.log(np.random.uniform(size=n_classes * size))).reshape(size, n_classes)
    dependent = (logits[independent] + gumbels).argmax(1)
    independent_values = categories[independent]
    dependent_values = categories[dependent]
    return pd.DataFrame({"x": independent_values, "y": dependent_values}).astype(str)


def create_multidimensional_gaussian(dimensions: int, size: int, prefix: str = "x") -> pd.DataFrame:
    """Draw `size` samples from a `dimensions`-dimensional standard gaussian. """
    z = np.random.randn(size, dimensions).astype(np.float32)
    columns = ["{}_{}".format(prefix, i) for i in range(dimensions)]
    df = pd.DataFrame(z, columns=columns)
    return df


def create_multidimensional_categorical(dimensions: int, n_classes: int, size: int, prefix: str = "x") -> pd.DataFrame:
    """Draw `size` samples of `dimensions` uniform, independent categorical random variables. """
    z = np.random.choice(list(map(str, range(n_classes))), dimensions * size).reshape(size, dimensions)
    columns = [f"{prefix}_{i}" for i in range(dimensions)]
    df = pd.DataFrame(z, columns=columns)
    return df


def create_multidimensional_correlated_categorical(n_classes: int, dimensions: int, sd: float = 1, size: int = 10000):
    """
    Create a dataset with `dimensions` dependent categorical columns each taking one of `n_classes`
    possible values.

    The first column is uniform distributed. The other columns are drawn sequentially by choosing one of
    the previous columns uniformly at random and sampling the current column's  value dependent on it.
    To specify the conditional distributions, for each of the `n_classes` possible values of the conditioning
     column, a vector of logits is drawn from `N(0, sd * I_{n_classes, n_classes})`.
    """
    logits = sd * np.random.randn(dimensions * n_classes * n_classes).reshape(dimensions, n_classes, n_classes)
    categories = np.array([f"v_{i}" for i in list(range(n_classes))])
    independent = np.random.choice(list(range(n_classes)), size=size)
    independent_values = categories[independent]
    gumbels = -np.log(
        -np.log(np.random.uniform(size=dimensions * size * n_classes))).reshape(dimensions, size, n_classes)
    val_list, val_dict = [independent], {"x_0": independent_values}
    col_ids = list(range(dimensions))
    for j in range(1, dimensions):
        parent_idx = choice(col_ids[:j])
        parent_vals = val_list[parent_idx]
        local_logits, local_gumbels = logits[j], gumbels[j]
        dependent = (local_logits[parent_vals] + local_gumbels).argmax(-1)
        val_list.append(dependent)
        dependent_values = categories[dependent]
        val_dict[f"x_{j}"] = dependent_values
    return pd.DataFrame(val_dict).astype(str)


def create_multidimensional_mixed(
        continuous_dim: int, categorical_dim: int, n_classes: int, size: int
) -> pd.DataFrame:
    """
    Draw `size` samples from a `continuous_dim`-dimensional standard gaussian and `categorical_dim` uniform,
    independent categorical random variables. The categorical column takes one of `n_classes` possible values.
    """
    continuous = create_multidimensional_gaussian(dimensions=continuous_dim, size=size, prefix="x")
    categorical = create_multidimensional_categorical(dimensions=categorical_dim, n_classes=n_classes,
                                                      size=size, prefix="y")
    return pd.concat([continuous, categorical], axis=1)


def sample_cat_to_cat(parent_values: np.array, sd: float, size: int, n_classes: int):
    logits = sd * np.random.randn(n_classes * n_classes).reshape(n_classes, n_classes)
    gumbels = -np.log(-np.log(np.random.uniform(size=size * n_classes))).reshape(size, n_classes)
    return (logits[parent_values] + gumbels).argmax(-1)


def sample_cat_to_cont(parent_values: np.array, size: int, n_classes: int, sd: float, prior_sd: float):
    means = prior_sd * np.random.randn(n_classes).astype(np.float32)
    sample_means = means[parent_values]
    zs = np.random.randn(size)
    return (sample_means + sd * zs).astype(np.float32)


def sample_cont_to_cont(parent_values: np.array, size: int, prior_sd: float, noise_sd: float):
    weight = gauss(mu=0, sigma=prior_sd)
    bias = gauss(mu=0, sigma=prior_sd)
    noise = np.random.randn(size)
    return (weight * parent_values + bias + noise_sd * noise).astype(np.float32)


def sample_cont_to_cat(parent_values: np.array, size: int, prior_sd: float, n_classes: int):
    weights, biases = prior_sd * np.random.randn(1, n_classes), prior_sd * np.random.randn(1, n_classes)
    class_weights = biases + np.multiply(np.expand_dims(parent_values, -1), weights)
    gumbels = -np.log(-np.log(np.random.uniform(size=size * n_classes))).reshape(size, n_classes)
    return (class_weights + gumbels).argmax(-1)


def create_multidimensional_correlated_mixed(continuous_dim: int, categorical_dim: int, n_classes: int,
                                             prior_sd: float, categorical_sd: float, cont_sd: float,
                                             size: int = 10000):
    """
    Create a dataset with `continuous_dim` continuous and `categorical_dim` categorical columns
    all of which are dependent. All of the categorical columns take one of `n_classes` possible
    values.

    The first column is always categorical and drawn from a uniform distribution over `n_classes`
    possible values. The other columns are drawn sequentially by choosing one of
    the previous columns uniformly at random and sampling the current column's  value dependent on it.

    If the current column is categorical and the conditioning column is categorical, it is drawn as in
    `create_correlated_categoricals`. If the conditioning column is continuous, the current column is
    sampled from a logistic regression model whose weights are biases are drawn from
    `N(0, prior_sd * I_{n_classes, n_classes})`.

    If the current column if continuous and the conditioning column is categorical, it is
    drawn as in `create_mixed_continuous_categorical`. If the conditioning column is continuous,
    it is drawn from a regression model with weights and biases drawn from `N(0, prior_sd * I_{n_classes, n_classes})`
    and noise drawn from `N(0, cont_sd * I_{n_classes, n_classes})`
    """
    categories = np.array([f"v_{i}" for i in list(range(n_classes))])
    kinds = continuous_dim * ["continuous"] + (categorical_dim - 1) * ["categorical"]
    shuffle(kinds)
    kinds.insert(0, "categorical")
    independent = np.random.choice(list(range(n_classes)), size=size)
    independent_values = categories[independent]
    val_list, val_dict = [independent], {"x_0": independent_values}
    col_ids = list(range(continuous_dim + categorical_dim))
    for j in range(1, continuous_dim + categorical_dim):
        kind = kinds[j]
        parent_idx = choice(col_ids[:j])
        parent_kind = kinds[parent_idx]
        parent_vals = val_list[parent_idx]
        if parent_kind == "categorical":
            if kind == "categorical":
                value = sample_cat_to_cat(parent_values=parent_vals, size=size, n_classes=n_classes,
                                          sd=categorical_sd)
            else:
                value = sample_cat_to_cont(parent_values=parent_vals, size=size, n_classes=n_classes,
                                           prior_sd=prior_sd, sd=cont_sd)
        else:
            if kind == "categorical":
                value = sample_cont_to_cat(parent_values=parent_vals, size=size, n_classes=n_classes,
                                           prior_sd=prior_sd)
            else:
                value = sample_cont_to_cont(parent_values=parent_vals, size=size, prior_sd=prior_sd,
                                            noise_sd=cont_sd)
        val_list.append(value)
        if kind == "categorical":
            val_dict[f"x_{j}"] = categories[value]
        else:
            val_dict[f"x_{j}"] = value
    return pd.DataFrame(val_dict)


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


def continuous_auto_regressive(phi, c, sd):
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


def categorical_auto_regressive(n_classes, sd):
    """
    A first-order Markov model with categorical values.

    The transition probabilities are specified by sampling
    the logits for each row from a multivariate normal distribution with
    zero mean and a diagonal covariance matrix with standard deviation
    `sd`.

    :param n_classes: [int] the number of categories
    :param sd: [float] standard deviation of prior on logits
    """
    categories = np.array([f"v_{i}" for i in list(range(n_classes))])
    logits = sd * np.random.randn(n_classes * n_classes).reshape(n_classes, n_classes)

    def out_func(times):
        size = times.shape[0]
        gumbels = -np.log(-np.log(np.random.uniform(size=size * n_classes))).reshape(size, n_classes)
        out_list = [np.random.randint(n_classes)]
        for i in range(1, size):
            prev_idx = out_list[-1]
            out_idx = (logits[prev_idx] + gumbels[prev_idx]).argmax(-1)
            out_list.append(out_idx)
        return categories[out_list]
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
