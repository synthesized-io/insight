import numpy as np
import pandas as pd
from typing import List, Dict
from scipy.stats import ks_2samp
from scipy.stats import spearmanr
from statsmodels.formula.api import mnlogit, ols
from statsmodels.tsa.stattools import acf, pacf

import synthesized
from synthesized.testing.evaluation import Evaluation


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
def continuous_correlation(x, y):
    return np.corrcoef(x, y)[0, 1]


def continuous_rsquared(x, y):
    return continuous_correlation(x, y)**2


def ordered_correlation(x, y):
    return spearmanr(x, y).correlation


def ordered_rsquared(x, y):
    return ordered_correlation(x, y)**2


# --- Association between categorical and categorical
def categorical_logistic_rsquared(x, y):
    temp_df = pd.DataFrame({"x": x, "y": y})
    model = mnlogit("y ~ C(x)", data=temp_df).fit(method="cg", disp=0)
    return model.prsquared


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


# -- training functions
def train_and_synthesize(data: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame,
                         evaluation: Evaluation, evaluation_name: str):
    synthesizer_class = evaluation.configs[evaluation_name]["synthesizer_class"]
    assert synthesizer_class in {"HighDimSynthesizer", "SeriesSynthesizer"}
    synthesizer_constructor = getattr(synthesized, synthesizer_class)
    loss_history: List[Dict[str, float]] = list()

    def callback(synth, iteration, losses):
        if len(loss_history) == 0:
            loss_history.append(losses)
        else:
            loss_history.append({name: losses[name] for name in loss_history[0]})

    with synthesizer_constructor(df=data, **evaluation.configs[evaluation_name]['params']) as synthesizer:
        synthesizer.learn(
            df_train=train, num_iterations=evaluation.configs[evaluation_name]['num_iterations'],
            callback=callback, callback_freq=100
        )
        if synthesizer_class == "HighDimSynthesizer":
            synthesized_data = synthesizer.synthesize(num_rows=len(test))
        else:
            series_lengths = data.groupby(
                evaluation.configs[evaluation_name]["params"]["identifier_label"]).count().to_numpy().transpose()
            series_lengths = list(series_lengths[0])
            synthesized_data = synthesizer.synthesize(series_lengths=series_lengths)
        return synthesizer, synthesized_data, loss_history
