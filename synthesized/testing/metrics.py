"""Generic metrics for various types/combinations of values."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

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
