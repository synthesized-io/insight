from typing import Tuple, Dict, List, Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import acf, pacf

from .plotting import set_plotting_style, plot_continuous_time_series, plot_categorical_time_series, \
    plot_cross_correlations
from ..metadata import DataFrameMeta
from ..insight import metrics
from ..insight.dataset import categorical_or_continuous_values, format_time_series

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

        self.identifiers = df_meta.id_value.identifiers

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

    def autocorrelation_diff_plot_seaborn(self, max_lag: int = 100) -> None:
        """Plot autocorrelation.

        Args:
            max_lag: A max lag
        """

        # for synthetic data at the moment, TODO for real data
        # how do we detect time column?
        def autocorrelation(h, data, mean, n, c0):
            return ((data[:n - h] - mean) *
                    (data[h:] - mean)).sum() / float(n) / c0

        n_orig, n_synth = len(self.df_orig), len(self.df_synth)
        original_data, synthetic_data = np.asarray(self.df_orig), np.asarray(self.df_synth)

        mean_orig, mean_synth = np.mean(original_data), np.mean(synthetic_data)
        c0_orig = np.sum((original_data - mean_orig) ** 2) / float(n_orig)
        c0_synth = np.sum((synthetic_data - mean_synth) ** 2) / float(n_synth)

        n = min(n_orig, n_synth, max_lag)
        x = np.arange(n) + 1

        y_orig = [autocorrelation(loc, original_data[:n], mean_orig, n, c0_orig) for loc in x]
        y_synth = [autocorrelation(loc, synthetic_data, mean_synth, n_synth, c0_synth) for loc in x]

        sns.set(style='whitegrid')

        data = pd.DataFrame({'Original': y_orig, 'Synthetic': y_synth})
        sns.lineplot(data=data, palette=[COLOR_SYNTH, COLOR_ORIG], linewidth=2.5)
        return mean_squared_error(y_orig, y_synth)

    def get_avg_fn(self, df, col, unique_ids: List, fn, nlags=40):
        distance = []
        if len(unique_ids) > nlags:
            for i in unique_ids:
                col_test = pd.to_numeric(df.loc[df[self.identifier] == i, col], errors='coerce').dropna().values
                if len(col_test) > 1:
                    distance.append(np.mean(fn(col_test, nlags=nlags)))
        else:
            col_test = pd.to_numeric(df[col], errors='coerce').dropna().values
            if len(col_test) > nlags:
                distance.append(np.mean(fn(col_test, nlags=nlags)))

        return distance

    def show_autocorrelation_distances(self, nlags: int = 40, plot_results: bool = True):
        """Plot a barplot with ACF-distances between original and synthetic columns."""
        result = []
        for col in self.continuous:

            acf_distance_orig = self.get_avg_fn(self.df_orig, col, unique_ids=self.unique_ids_orig, fn=acf, nlags=nlags)
            acf_distance_synth = self.get_avg_fn(self.df_synth, col, unique_ids=self.unique_ids_synth, fn=acf,
                                                 nlags=nlags)

            if len(acf_distance_synth) == 0 or len(acf_distance_synth) == 0:
                continue

            acf_distance = np.abs(np.mean(acf_distance_orig) - np.mean(acf_distance_synth))
            result.append({'column': col, 'distance': acf_distance})

        if len(result) == 0:
            return 0., 0.

        df = pd.DataFrame.from_records(result)

        acf_dist_max = df['distance'].max()
        acf_dist_avg = df['distance'].mean()

        print("Max ACF distance:", acf_dist_max)
        print("Average ACF distance:", acf_dist_avg)

        if plot_results:
            plt.figure(figsize=(8, np.ceil(len(df) / 2)))
            g = sns.barplot(y='column', x='distance', data=df)
            g.set_xlim(0.0, 1.0)
            plt.title('ACF Distances')
            plt.show()

        return acf_dist_max, acf_dist_avg

    def show_partial_autocorrelation_distances(self, nlags=40):
        """Plot a barplot with PACF-distances between original and synthetic columns."""
        result = []
        for col in self.continuous_cols:

            pacf_distance_orig = self.get_avg_fn(self.df_test, col, unique_ids=self.unique_ids_orig, fn=pacf,
                                                 nlags=nlags)
            pacf_distance_synth = self.get_avg_fn(self.df_synth, col, unique_ids=self.unique_ids_synth, fn=pacf,
                                                  nlags=nlags)

            if len(pacf_distance_synth) == 0 or len(pacf_distance_synth) == 0:
                continue

            pacf_distance = np.abs(np.mean(pacf_distance_orig) - np.mean(pacf_distance_synth))
            result.append({'column': col, 'distance': pacf_distance})

        if len(result) == 0:
            return 0., 0.

        df = pd.DataFrame.from_records(result)

        pacf_dist_max = df['distance'].max()
        pacf_dist_avg = df['distance'].mean()

        print("Max PACF distance:", pacf_dist_max)
        print("Average PACF distance:", pacf_dist_avg)

        plt.figure(figsize=(8, np.ceil(len(df) / 2)))
        g = sns.barplot(y='column', x='distance', data=df)
        g.set_xlim(0.0, 1.0)
        plt.title('PACF Distances')
        plt.show()

        return pacf_dist_max, pacf_dist_avg

    def show_transition_distances(self, plot_results: bool = True):
        """Plot a barplot with ACF-distances between original and synthetic columns."""
        result = []
        for col in self.categorical:
            val2idx = {v: i for i, v in enumerate(np.unique(self.df_orig[col]))}
            # ORIGINAL DATA
            t_orig = np.zeros((len(val2idx), len(val2idx)))
            k = 0
            if self.identifier:
                for i in self.unique_ids_orig:
                    t_orig += transition_matrix(
                        self.df_orig.loc[self.df_orig[self.identifier] == i, col], val2idx=val2idx)[0]
                    k += 1
                t_orig /= k
            else:
                t_orig += transition_matrix(
                    self.df_orig[col], val2idx=val2idx)[0]

            # Convert to dataframe
            t_orig = pd.DataFrame(t_orig, columns=list(np.unique(self.df_orig[col])))
            t_orig.index = np.unique(self.df_orig[col])

            # SYNTHESIZED DATA
            t_synth = np.zeros((len(val2idx), len(val2idx)))
            k = 0
            if self.identifier:
                for i in self.unique_ids_synth:
                    t_synth += transition_matrix(
                        self.df_synth.loc[self.df_synth[self.identifier] == i, col], val2idx=val2idx)[0]
                    k += 1
                t_synth /= k
            else:
                t_synth += transition_matrix(
                    self.df_synth[col], val2idx=val2idx)[0]

            # Convert to dataframe
            t_synth = pd.DataFrame(t_synth, columns=list(np.unique(self.df_synth[col])))
            t_synth.index = np.unique(self.df_synth[col])

            # Generate a custom diverging colormap
            cmap = sns.diverging_palette(220, 10, as_cmap=True)

            if plot_results:
                f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 7), sharex=True, sharey=True)

                # Draw the heatmap with the mask and correct aspect ratio
                sns.heatmap(t_orig, cmap=cmap, vmin=0.0, vmax=1.0, center=0,
                            square=True, linewidths=.5, cbar=False, ax=ax1, annot=True, fmt='.2f')
                sns.heatmap(t_synth, cmap=cmap, vmin=0.0, vmax=1.0, center=0,
                            square=True, linewidths=.5, cbar=False, ax=ax2, annot=True, fmt='.2f')
                sns.heatmap(abs(t_orig - t_synth), cmap=cmap, vmin=0.0, vmax=1.0, center=0,
                            square=True, linewidths=.5, cbar=False, ax=ax3, annot=True, fmt='.2f')
                ax2.set_ylim(ax2.get_ylim()[0] + .5, ax2.get_ylim()[1] - .5)

                ax1.set_title(col + ' - Transition Distances (Original)')
                ax2.set_title(col + ' - Transition Distances (Synthesized)')
                ax3.set_title(col + ' - Transition Distances (Difference)')
                plt.show()

            result.append({'column': col, 'distance': abs(t_orig - t_synth).mean().mean()})

        if len(result) == 0:
            return 0., 0.

        df = pd.DataFrame.from_records(result)

        dist_max = df['distance'].max()
        dist_avg = df['distance'].mean()

        print("Max Transition distance:", dist_max)
        print("Average Transition distance:", dist_avg)

        if plot_results:
            plt.figure(figsize=(8, np.ceil(len(df) / 2)))
            g = sns.barplot(y='column', x='distance', data=df)
            g.set_xlim(0.0, 1.0)
            plt.title('ACF Distances')
            plt.show()

        return dist_max, dist_avg


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
