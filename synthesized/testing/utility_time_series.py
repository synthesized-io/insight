from typing import Tuple, Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, normalized_mutual_info_score
from statsmodels.formula.api import ols
from statsmodels.tsa.stattools import acf, pacf

from ..common.synthesizer import Synthesizer
from ..values import CategoricalValue
from ..values import ContinuousValue
from ..values import DateValue
from ..values import DecomposedContinuousValue
from ..values import NanValue
from ..values import Value
from ..testing import metrics as eval_metrics
from ..insight import metrics
from ..insight.modelling import predictive_modelling_comparison
from .utility import DisplayType, COLOR_ORIG, COLOR_SYNTH


class TimeSeriesUtilityTesting:
    """A universal set of utilities that let you to compare quality of original vs synthetic data."""

    def __init__(self,
                 synthesizer: Synthesizer,
                 df_orig: pd.DataFrame,
                 df_test: pd.DataFrame,
                 df_synth: pd.DataFrame,
                 identifier=None):
        """Create an instance of UtilityTesting.

        Args:
            synthesizer: A synthesizer instance.
            df_orig: A DataFrame with original data that was used for training
            df_test: A DataFrame with hold-out original data
            df_synth: A DataFrame with synthetic data
        """
        self.df_orig = df_orig.copy()
        self.df_test = df_test.copy()
        self.df_synth = df_synth.copy()

        self.df_orig_encoded = synthesizer.preprocess(df=df_orig)
        self.df_test_encoded = synthesizer.preprocess(df=df_test)
        self.df_synth_encoded = synthesizer.preprocess(df=df_synth)

        self.date_cols: List[str] = []
        self.continuous_cols: List[str] = []
        self.categorical_cols: List[str] = []

        self.display_types: Dict[str, DisplayType] = {}
        for value in synthesizer.get_values():
            if isinstance(value, NanValue):
                value = value.value

            if isinstance(value, DateValue):
                self.df_orig[value.name] = pd.to_datetime(self.df_orig[value.name])
                self.df_test[value.name] = pd.to_datetime(self.df_test[value.name])
                self.df_synth[value.name] = pd.to_datetime(self.df_synth[value.name])
                self.date_cols.append(value.name)
            elif isinstance(value, ContinuousValue) or isinstance(value, DecomposedContinuousValue):
                self.display_types[value.name] = DisplayType.CONTINUOUS
                self.continuous_cols.append(value.name)
            elif isinstance(value, CategoricalValue):
                self.display_types[value.name] = DisplayType.CATEGORICAL
                self.categorical_cols.append(value.name)
        self.value_by_name: Dict[str, Value] = {}
        for v in synthesizer.get_values():
            self.value_by_name[v.name] = v

        # Identifiers (only for Time-Series)
        self.identifier = identifier
        if identifier:
            self.unique_ids_orig = self.df_orig[identifier].unique()
            self.unique_ids_synth = self.df_synth[identifier].unique()
        else:
            self.unique_ids_orig = []
            self.unique_ids_synth = []

        # Set the style of plots
        plt.style.use('seaborn')
        mpl.rcParams["axes.facecolor"] = 'w'
        mpl.rcParams['grid.color'] = 'grey'
        mpl.rcParams['grid.alpha'] = 0.1

        mpl.rcParams['axes.linewidth'] = 0.3
        mpl.rcParams['axes.edgecolor'] = 'grey'

        mpl.rcParams['axes.spines.right'] = True
        mpl.rcParams['axes.spines.top'] = True

    def _filter_column_data_types(self):
        categorical, continuous = [], []
        for name, dtype in self.display_types.items():
            if dtype == DisplayType.CONTINUOUS:
                continuous.append(name)
            elif dtype == DisplayType.CATEGORICAL:
                categorical.append(name)
        return categorical, continuous

    def show_auto_associations(self, figsize: Tuple[float, float] = (14, 50), cols: int = 2, max_order=30):
        fig = plt.figure(figsize=figsize)
        for i, (col, dtype) in enumerate(self.display_types.items()):
            ax = fig.add_subplot(len(self.display_types), cols, i + 1)
            n = list(range(1, max_order + 1))
            original_auto = eval_metrics.calculate_auto_association(dataset=self.df_test, col=col, max_order=30)
            synth_auto = eval_metrics.calculate_auto_association(dataset=self.df_synth, col=col, max_order=30)
            ax.stem(n, original_auto, 'b', markerfmt='bo', label="Original")
            ax.stem(n, synth_auto, 'g', markerfmt='go', label="Synthetic")
            ax.set_title(label=col)
            ax.legend()

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

        n_orig, n_synth = len(self.df_test), len(self.df_synth)
        original_data, synthetic_data = np.asarray(self.df_test), np.asarray(self.df_synth)

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
        for col in self.continuous_cols:

            acf_distance_orig = self.get_avg_fn(self.df_test, col, unique_ids=self.unique_ids_orig, fn=acf, nlags=nlags)
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

    def show_series(self, num_series=10, figsize: Tuple[float, float] = None, share_axis=False, share_ids=False):

        if not figsize:
            figsize = (14, 10 * len(self.display_types))

        if self.identifier:
            num_series = min(num_series, len(np.unique(self.unique_ids_orig)), len(np.unique(self.unique_ids_synth)))
            identifiers_orig = np.random.choice(self.unique_ids_orig, num_series, replace=False)
            if share_ids:
                identifiers_synth = identifiers_orig
            else:
                identifiers_synth = np.random.choice(self.unique_ids_synth, num_series, replace=False)

        fig = plt.figure(figsize=figsize)
        for i in range(len(self.continuous_cols)):
            col = self.continuous_cols[i]

            # Original
            ax = fig.add_subplot(2 * len(self.continuous_cols), 2, 2 * i + 1)

            if self.identifier:
                for idf in identifiers_orig:
                    x = pd.to_numeric(self.df_orig.loc[self.df_orig[self.identifier] == idf, col], errors='coerce'
                                      ).dropna().values
                    if len(x) > 1:
                        ax.plot(range(len(x)), x, label=idf)
                ax.legend()
            else:
                x = pd.to_numeric(self.df_orig[col], errors='coerce').dropna().values
                if len(x) > 1:
                    ax.plot(range(len(x)), x)
            ax.set_title(col + ' (Original)')

            # Synthesized
            if share_axis:
                ax = fig.add_subplot(2 * len(self.continuous_cols), 2, 2 * i + 2, sharex=ax, sharey=ax)
            else:
                ax = fig.add_subplot(2 * len(self.continuous_cols), 2, 2 * i + 2)
            if self.identifier:
                for idf in identifiers_synth:
                    x = self.df_synth.loc[self.df_synth[self.identifier] == idf, col].dropna().values
                    ax.plot(range(len(x)), x, label=idf)
            else:
                x = self.df_synth[col].dropna().values
                ax.plot(range(len(x)), x)
            ax.legend()
            ax.set_title(col + ' (Synthesized)')

        plt.tight_layout()
        plt.show()

    def show_transition_distances(self, plot_results: bool = True):
        """Plot a barplot with ACF-distances between original and synthetic columns."""
        result = []
        for col in self.categorical_cols:
            val2idx = {v: i for i, v in enumerate(np.unique(self.df_test[col]))}
            # ORIGINAL DATA
            t_orig = np.zeros((len(val2idx), len(val2idx)))
            k = 0
            if self.identifier:
                for i in self.unique_ids_orig:
                    t_orig += eval_metrics.transition_matrix(
                        self.df_test.loc[self.df_test[self.identifier] == i, col], val2idx=val2idx)[0]
                    k += 1
                t_orig /= k
            else:
                t_orig += eval_metrics.transition_matrix(
                    self.df_test[col], val2idx=val2idx)[0]

            # Convert to dataframe
            t_orig = pd.DataFrame(t_orig, columns=list(np.unique(self.df_test[col])))
            t_orig.index = np.unique(self.df_test[col])

            # SYNTHESIZED DATA
            t_synth = np.zeros((len(val2idx), len(val2idx)))
            k = 0
            if self.identifier:
                for i in self.unique_ids_synth:
                    t_synth += eval_metrics.transition_matrix(
                        self.df_synth.loc[self.df_synth[self.identifier] == i, col], val2idx=val2idx)[0]
                    k += 1
                t_synth /= k
            else:
                t_synth += eval_metrics.transition_matrix(
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
