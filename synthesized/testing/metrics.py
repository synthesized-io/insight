"""Generic metrics for various types/combinations of values."""
import logging
from typing import List, Union, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

from ..insight import metrics
from ..insight.dataset import categorical_or_continuous_values

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s :: %(levelname)s :: %(message)s', level=logging.INFO)

MAX_SAMPLE_DATES = 2500
NUM_UNIQUE_CATEGORICAL = 100
NAN_FRACTION_THRESHOLD = 0.25
NON_NAN_COUNT_THRESHOLD = 500
CATEGORICAL_THRESHOLD_LOG_MULTIPLIER = 2.5


# -- Measures of association for different pairs of data types
def calculate_auto_association(dataset: pd.DataFrame, col: str, max_order: int):
    variable = dataset[col].to_numpy()
    categorical, continuous = categorical_or_continuous_values(dataset[col])

    association: Optional[metrics.TwoColumnComparison] = None

    if len(categorical) > 0:
        association = metrics.earth_movers_distance
    elif len(continuous) > 0:
        association = metrics.kendell_tau_correlation

    if association is None:
        return None

    auto_associations = []
    for order in range(1, max_order+1):
        postfix = variable[order:]
        prefix = variable[:-order]
        df = pd.DataFrame({'prefix': prefix, 'postfix': postfix})
        auto_associations.append(association(df, 'prefix', 'postfix'))
    return np.array(auto_associations)


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


def transition_matrix(transitions: np.array, val2idx: Dict[int, Any] = None) -> Tuple[np.array, Dict[int, Any]]:
    if not val2idx:
        val2idx = {v: i for i, v in enumerate(np.unique(transitions))}

    n = len(val2idx)  # number of states
    M = np.zeros((n, n))

    for (v_i, v_j) in zip(transitions, transitions[1:]):
        M[val2idx[v_i], val2idx[v_j]] += 1

    for row in M:
        s = sum(row)
        if s > 0:
            row[:] = [f / s for f in row]

    return M, val2idx
