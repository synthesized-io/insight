"""Generic metrics for various types/combinations of values."""
import logging
from itertools import chain
from typing import List, Union, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ks_2samp, spearmanr, kendalltau
from statsmodels.formula.api import mnlogit
from statsmodels.formula.api import ols
from statsmodels.tsa.stattools import acf, pacf

from .util import categorical_emd

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s :: %(levelname)s :: %(message)s', level=logging.INFO)

MAX_SAMPLE_DATES = 2500
NUM_UNIQUE_CATEGORICAL = 100
MAX_PVAL = 0.05
NAN_FRACTION_THRESHOLD = 0.25
NON_NAN_COUNT_THRESHOLD = 500
CATEGORICAL_THRESHOLD_LOG_MULTIPLIER = 2.5


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


def get_cramers_v_matrix(df: pd.DataFrame, flattened: bool = False) -> Union[pd.DataFrame, np.array]:
    """Compute Cramer's V between all the columns in a given DataFrame and write result to a matrix"""
    columns = df.columns
    if flattened:
        cramers_v_matrix: Union[pd.DataFrame, np.array] = []
    else:
        cramers_v_matrix = pd.DataFrame(np.ones((len(columns), len(columns))), columns=columns)
        cramers_v_matrix.index = columns

    for i1 in range(len(columns)):
        c1 = columns[i1]
        for i2 in range(i1 + 1, len(columns)):
            c2 = columns[i2]
            cv = cramers_v(df[c1], df[c2])
            if flattened:
                cramers_v_matrix.append(cv)
            else:
                cramers_v_matrix[c1][c2] = cramers_v_matrix[c2][c1] = cramers_v(df[c1], df[c2])

    if flattened:
        return np.array(cramers_v_matrix)
    else:
        return cramers_v_matrix


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


def calculate_evaluation_metrics(df_orig: pd.DataFrame, df_synth: pd.DataFrame,
                                 column_names: Optional[List[str]] = None) -> Dict[str, List[float]]:
    """Calculate 'stop_metric' dictionary given two datasets. Each item in the dictionary will include a key
    (from self.stop_metric_name, allowed options are 'ks_dist', 'corr' and 'emd'), and a value (list of
    stop_metrics per column).

    Args
        df_orig: Original DataFrame.
        df_synth: Synthesized DataFrame.
        column_names: List of columns used to compute the 'break_metric'.

    Returns
        bool: True if criteria are met to stop learning.
    """
    df_orig = df_orig.copy()
    df_synth = df_synth.copy()

    if column_names is None:
        column_names_df: List[str] = df_orig.columns
    else:
        column_names_df = list(filter(lambda c: c in df_orig.columns, column_names))

    # Calculate distances for all columns
    ks_distances = []
    emd_distances = []
    corr_distances = []

    numerical_columns = []
    categorical_columns = []

    len_test = len(df_orig)
    for col in column_names_df:

        if df_orig[col].dtype.kind == 'f':
            col_test_clean = df_orig[col].dropna()
            col_synth_clean = df_synth[col].dropna()
            if len(col_test_clean) < len_test:
                logger.debug("Column '{}' contains NaNs. Computing KS distance with {}/{} samples"
                             .format(col, len(col_test_clean), len_test))
            ks_distance, _ = ks_2samp(col_test_clean, col_synth_clean)
            ks_distances.append(ks_distance)
            numerical_columns.append(col)

        elif df_orig[col].dtype.kind == 'i':
            if df_orig[col].nunique() < np.log(len(df_orig)) * CATEGORICAL_THRESHOLD_LOG_MULTIPLIER:
                logger.debug(
                    "Column '{}' treated as categorical with {} categories".format(col, df_orig[col].nunique()))
                emd_distance = categorical_emd(df_orig[col].dropna(), df_synth[col].dropna())
                emd_distances.append(emd_distance)
                categorical_columns.append(col)

            else:
                col_test_clean = df_orig[col].dropna()
                col_synth_clean = df_synth[col].dropna()
                if len(col_test_clean) < len_test:
                    logger.debug("Column '{}' contains NaNs. Computing KS distance with {}/{} samples"
                                 .format(col, len(col_test_clean), len_test))
                ks_distance, _ = ks_2samp(col_test_clean, col_synth_clean)
                ks_distances.append(ks_distance)

            numerical_columns.append(col)

        elif df_orig[col].dtype.kind in ('O', 'b'):

            # Try to convert to numeric
            col_num = pd.to_numeric(df_orig[col], errors='coerce')
            if col_num.isna().sum() / len(col_num) < NAN_FRACTION_THRESHOLD:
                df_orig[col] = col_num
                df_synth[col] = pd.to_numeric(df_synth[col], errors='coerce')
                numerical_columns.append(col)

            # if (is not sampling) and (is not date):
            elif (df_orig[col].nunique() <= np.log(len(df_orig)) * CATEGORICAL_THRESHOLD_LOG_MULTIPLIER) and \
                    np.all(pd.to_datetime(df_orig[col].sample(min(len(df_orig), MAX_SAMPLE_DATES)),
                                          errors='coerce').isna()):
                emd_distance = categorical_emd(df_orig[col].dropna(), df_synth[col].dropna())
                if not np.isnan(emd_distance):
                    emd_distances.append(emd_distance)
                categorical_columns.append(col)

    # Compute correlation distances
    for i in range(len(numerical_columns)):
        col_i = numerical_columns[i]
        for j in range(i + 1, len(numerical_columns)):
            col_j = numerical_columns[j]
            test_clean = df_orig[[col_i, col_j]].dropna()
            synth_clean = df_synth[[col_i, col_j]].dropna()

            if len(test_clean) > 0 and len(synth_clean) > 0:
                corr_orig, pvalue_orig = kendalltau(test_clean[col_i].values, test_clean[col_j].values)
                corr_synth, pvalue_synth = kendalltau(synth_clean[col_i].values, synth_clean[col_j].values)

                if pvalue_orig <= MAX_PVAL or pvalue_synth <= MAX_PVAL:
                    corr_distances.append(abs(corr_orig - corr_synth))

    # Compute Cramer's V distances
    cramers_v_distances = np.abs(get_cramers_v_matrix(df_orig[categorical_columns], flattened=True) -
                                 get_cramers_v_matrix(df_synth[categorical_columns], flattened=True))

    stop_metrics = {
        'ks_distances': ks_distances,
        'emd_distances': emd_distances,
        'corr_distances': corr_distances,
        'cramers_v_distances': cramers_v_distances
    }

    return stop_metrics


# -- global constants
default_metrics = {"avg_distance": mean_ks_distance}
association_dict = {"i": ordered_correlation, "f": continuous_correlation,
                    "O": categorical_logistic_rsquared}
