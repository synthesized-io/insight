"""Generic metrics for various types/combinations of values."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ks_2samp
from scipy.stats import spearmanr
from statsmodels.formula.api import ols
from statsmodels.tsa.stattools import acf, pacf
from itertools import chain
from typing import List, Union
from statsmodels.formula.api import mnlogit


def continuous_correlation(x: List[Union[int, float]], y: List[Union[int, float]]) -> float:
    """Correlation coefficient between two continuous variables."""
    return np.corrcoef(x, y)[0, 1]


def continuous_rsquared(x: List[Union[int, float]], y: List[Union[int, float]]) -> float:
    """Association between continuous and continuous"""
    return continuous_correlation(x, y)**2


def categorical_logistic_rsquared(x: list, y: list) -> float:
    """Nominal association between categorical and categorical"""
    temp_df = pd.DataFrame({"x": x, "y": y})
    model = mnlogit("y ~ C(x)", data=temp_df).fit(method="cg", disp=0)
    return model.prsquared


def contingency_table(x: list, y: list) -> sm.stats.Table:
    """Returns the contingency table of two categorical variables."""
    temp_df = pd.DataFrame({"x": x, "y": y})
    table = sm.stats.Table.from_data(temp_df)
    return table


def cramers_v(x: list, y: list) -> float:
    """Nominal association between categorical and categorical"""
    table = contingency_table(x, y)
    expected = table.fittedvalues.to_numpy()
    real = table.table
    r, c = real.shape
    n = np.sum(real)
    v = np.sum((real - expected) ** 2 / (expected * n * min(r - 1, c - 1))) ** 0.5

    return v


def normalized_contingency_residuals(x: list, y: list) -> pd.DataFrame:
    """Returns the associations between two categorical variables for a data set."""
    xcats = np.sort(list(set(x))).tolist()
    ycats = np.sort(list(set(y))).tolist()

    table = contingency_table(x, y)
    expected = table.fittedvalues.to_numpy() + 1.0
    real = table.table + 1.0

    values = ((real - expected) / expected).flatten()
    records = [(xcats[n // len(ycats)], ycats[n % len(ycats)], v) for n, v in enumerate(values)]
    x, y, resids = zip(*records)
    df = pd.DataFrame({'x': x, 'y': y, 'residuals': resids})
    return df


def normalized_contingency_residuals_diff(x1: list, y1: list, x2: list, y2: list) -> pd.DataFrame:
    """Returns the differences in association between two categorical variables for two data sets."""
    df1 = normalized_contingency_residuals(x1, y1)
    df2 = normalized_contingency_residuals(x2, y2)

    dict1 = {(df1['x'][n], df1['y'][n]): df1['residuals'][n] for n in range(len(df1))}
    dict2 = {(df2['x'][n], df2['y'][n]): df2['residuals'][n] for n in range(len(df2))}

    dict3 = {
        k: np.abs(dict1.get(k, 0.0) - dict2.get(k, 0.0))
        for k in set(chain(dict1.keys(), dict2.keys()))
    }

    xy, resids = zip(*dict3.items())
    x, y = zip(*xy)
    df = pd.DataFrame({'x': x, 'y': y, 'residuals': resids})
    return df


def max_contingency_residuals_diff(x1: list, y1: list, x2: list, y2: list) -> float:
    """Returns the largest difference in association between two categorical variables for two data sets."""
    return max(normalized_contingency_residuals_diff(x1, y1, x2, y2)['residuals'])


# -- Measures of association for different pairs of data types
def calculate_auto_association(dataset: pd.DataFrame, col: str, max_order: int):
    variable = dataset[col].to_numpy()
    association = association_dict[variable.dtype.kind]
    auto_associations = []
    for order in range(1, max_order+1):
        postfix = variable[order:]
        prefix = variable[:-order]
        auto_associations.append(association(x=prefix, y=postfix))
    return np.array(auto_associations)


# --- Association between continuous and continuous
def ordered_correlation(x, y):
    return spearmanr(x, y).correlation


def ordered_rsquared(x, y):
    return ordered_correlation(x, y)**2


# --- Association between continuous and categorical
def continuous_logistic_rsquared(x, y):
    temp_df = pd.DataFrame({"x": x, "y": y})
    model = mnlogit("y ~ x", data=temp_df).fit(method="cg", disp=0)
    return model.prsquared


# --- Association between categorical and continuous
def anova_rsquared(x, y):
    temp_df = pd.DataFrame({"x": x, "y": y})
    model = ols("y ~ C(x)", data=temp_df).fit(method="cg", disp=0)
    return model.rsquared


# -- Evaluation metrics
def max_correlation_distance(orig: pd.DataFrame, synth: pd.DataFrame):
    return np.abs((orig.corr() - synth.corr()).to_numpy()).max()


def max_autocorrelation_distance(orig: pd.DataFrame, synth: pd.DataFrame):
    floats = [col for dtype, col in zip(orig.dtypes.values, orig.columns.values)
              if dtype.kind == "f"]
    acf_distances = [np.abs((acf(orig[col], fft=True) - acf(synth[col], fft=True))).max()
                     for col in floats]
    return max(acf_distances)


def max_partial_autocorrelation_distance(orig: pd.DataFrame, synth: pd.DataFrame):
    floats = [col for dtype, col in zip(orig.dtypes.values, orig.columns.values)
              if dtype.kind == "f"]
    pacf_distances = [np.abs((pacf(orig[col]) - pacf(synth[col]))).max()
                      for col in floats]
    return max(pacf_distances)


def max_categorical_auto_association_distance(orig: pd.DataFrame, synth: pd.DataFrame, max_order=20):
    cats = [col for dtype, col in zip(orig.dtypes.values, orig.columns.values)
            if dtype.kind == "O"]
    cat_distances = [np.abs(calculate_auto_association(orig, col, max_order) -
                            calculate_auto_association(synth, col, max_order)).max()
                     for col in cats]
    return max(cat_distances)


def mean_squared_error_closure(col, baseline: float = 1):
    def mean_squared_error(orig: pd.DataFrame, synth: pd.DataFrame):
        return ((orig[col].to_numpy() - synth[col].to_numpy())**2).mean()/baseline
    return mean_squared_error


def mean_ks_distance(orig: pd.DataFrame, synth: pd.DataFrame):
    distances = [ks_2samp(orig[col], synth[col])[0] for col in orig.columns]
    return np.mean(distances)


def rolling_mse_asof(sd, time_unit=None):
    """
    Calculate the mean-squared error between the "x" values of the original and synthetic
    data. The sets of times may not be identical so we use "as of" (last observation rolled
    forward) to interpolate between the times in the two datasets.

    The dates are also optionally truncated to some unit following the syntax for the pandas
    `.floor` function.

    :param sd: [float] error standard deviation
    :param time_unit: [str] the time unit to round to. See documentation for pandas `.floor` method.
    :return: [(float, float)] MSE and MSE/(2*error variance)
    """
    # truncate date
    def mse_function(orig, synth):
        if time_unit is not None:
            synth.t = synth.t.dt.floor(time_unit)
            orig.t = orig.t.dt.floor(time_unit)

        # join datasets
        joined = pd.merge_asof(orig[["t", "x"]], synth[["t", "x"]], on="t")

        # calculate metrics
        mse = ((joined.x_x - joined.x_y) ** 2).mean()
        mse_eff = mse / (2 * sd ** 2)

        return mse_eff
    return mse_function


# -- global constants
default_metrics = {"avg_distance": mean_ks_distance}
association_dict = {"i": ordered_correlation, "f": continuous_correlation,
                    "O": categorical_logistic_rsquared}
