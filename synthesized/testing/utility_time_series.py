from typing import Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting import set_plotting_style, plot_continuous_time_series, plot_categorical_time_series, \
    plot_cross_correlations, plot_series
from ..metadata import DataFrameMeta
from ..insight import metrics
from ..insight.metrics import ColumnVector
from ..insight.dataset import categorical_or_continuous_values

COLOR_ORIG = '#1C5D7A'
COLOR_SYNTH = '#801761'


class TimeSeriesUtilityTesting:
    """A universal set of utilities that let you to compare quality of original vs synthetic data."""

    def __init__(self, df_meta: DataFrameMeta, df_orig: pd.DataFrame, df_synth: pd.DataFrame,
                 forecast_from=None):
        """Create an instance of UtilityTesting.

        Args:
            df_meta: DataFrame meta information.
            df_orig: A DataFrame with original data that was used for training
            df_synth: A DataFrame with synthetic data
        """
        self.df_meta = df_meta
        self.id_index = df_meta.id_index_name
        self.time_index = df_meta.time_index_name

        self.df_orig = self.df_meta.set_indices(df_orig.copy())
        self.df_synth = self.df_meta.set_indices(df_synth.copy())
        self.forecast_from = forecast_from

        categorical, continuous = categorical_or_continuous_values(self.df_meta)

        self.categorical, self.continuous = [v.name for v in categorical], [v.name for v in continuous]
        self.plotable_values = self.categorical + self.continuous

        # Identifiers (only for Time-Series)
        self.unique_ids_orig = self.df_orig.index.get_level_values(self.id_index).unique()
        self.unique_ids_synth = self.df_synth.index.get_level_values(self.id_index).unique()

        self.identifiers = df_meta.id_value.identifiers if df_meta.id_value is not None else []

        # Set the style of plots
        set_plotting_style()

    def show_series(self):

        rows = len(self.plotable_values)
        width = 14
        height = 8 * rows + 3
        figsize = (width, height)

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(nrows=rows, ncols=1, left=.05, bottom=1/height, right=.95, top=(height-2)/height,
                              wspace=0, hspace=.2)

        for i in range(len(self.continuous)):
            col = self.continuous[i]
            ax = fig.add_subplot(gs[i])
            ax.set_title(col, fontsize=16, pad=32)
            ax.set_axis_off()
            plot_continuous_time_series(self.df_orig, self.df_synth, col, forecast_from=self.forecast_from,
                                        identifiers=self.identifiers, ax=ax)

        for i in range(len(self.categorical)):
            col = self.categorical[i]
            ax = fig.add_subplot(gs[i+len(self.continuous)])
            ax.set_title(col, fontsize=16, pad=32)
            ax.set_axis_off()
            plot_categorical_time_series(self.df_orig, self.df_synth, col=col, identifiers=self.identifiers, ax=ax)

        plt.suptitle('Time-Series', fontweight='bold', fontsize=24)
        plt.show()

    def show_continuous_column_vector(self, metric_vector: ColumnVector, col: str, id_: str, **kwargs):
        xs_orig = self.df_orig.xs(id_)
        xs_synth = self.df_synth.xs(id_)

        sr_orig = metric_vector(sr=xs_orig[col], **kwargs)
        sr_synth = metric_vector(sr=xs_synth[col], **kwargs)

        rows = 1
        width = 20
        height = 8 * rows + 3
        figsize = (width, height)
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()

        plot_series(sr_orig, ax=ax, color=COLOR_ORIG)
        sr_orig_avg = np.nanmean(sr_orig.values)
        sr_orig[sr_orig.notna()] = sr_orig_avg
        plot_series(sr_orig, ax=ax, color=COLOR_ORIG, linestyle='dashed')

        plot_series(sr_synth, ax=ax, color=COLOR_SYNTH)
        sr_synth_avg = np.nanmean(sr_synth.values)
        sr_synth[sr_synth.notna()] = sr_synth_avg
        plot_series(sr_synth, ax=ax, color=COLOR_SYNTH, linestyle='dashed')

        plt.suptitle(f'{metric_vector.name} (Orig Avg: {sr_orig_avg:.3f} Synth Avg: {sr_synth_avg:.3f})',
                     fontweight='bold', fontsize=24)
        plt.show()

    def show_auto_associations(self, max_order=30):
        rows = len(self.continuous)
        width = 20
        height = 8 * rows + 3
        figsize = (width, height)
        fig = plt.figure(figsize=figsize)

        gs = fig.add_gridspec(nrows=rows, ncols=1, left=.05, bottom=1 / height, right=.95, top=(height - 2) / height,
                              wspace=0, hspace=.2)
        for i, col in enumerate(self.continuous):
            ax = fig.add_subplot(gs[i])
            ax.set_title(col, fontsize=16, pad=32)
            ax.set_axis_off()
            plot_cross_correlations(self.df_orig, self.df_synth, col, identifiers=self.identifiers,
                                    max_order=max_order, ax=ax)
        plt.suptitle('Cross-correlations', fontweight='bold', fontsize=24)
        plt.show()


# -- Measures of association for different pairs of data types
def calculate_auto_association(dataset: pd.DataFrame, col: str, max_order: int):
    variable = dataset[col].to_numpy()
    categorical, continuous = categorical_or_continuous_values(dataset[col])

    if len(categorical) > 0:
        def association(df_pre, df_post, col):
            return metrics.earth_movers_distance(df_pre, df_post, col)
    elif len(continuous) > 0:
        def association(df_pre, df_post, col):
            df = pd.DataFrame({'pre': df_pre[col], 'post': df_post[col]})
            return metrics.kendell_tau_correlation(df, 'pre', 'post')
    else:
        return None

    auto_associations = []
    for order in range(1, max_order+1):
        postfix = variable[order:]
        prefix = variable[:-order]
        df_pre, df_post = pd.DataFrame({col: prefix}), pd.DataFrame({col: postfix})
        auto_associations.append(association(df_pre, df_post, col))
    return np.array(auto_associations)


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
