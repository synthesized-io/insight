# -*- coding: utf-8 -*-
#
# Produced by: Synthesized Ltd
#
# The testing environment
"""This module contains tools for utility testing."""
from enum import Enum
import logging
from typing import Tuple, Dict, List, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..common.synthesizer import Synthesizer
from ..insight import metrics
from ..insight.metrics import TwoColumnComparison, TwoColumnComparisonMatrix
from ..insight.metrics import TwoColumnMetric, TwoColumnMetricMatrix
from ..insight.metrics import ColumnComparison, ColumnComparisonVector
from ..insight.modelling import predictive_modelling_comparison

COLOR_ORIG = '#00AB26'
COLOR_SYNTH = '#2794F3'

logger = logging.getLogger(__name__)


class DisplayType(Enum):
    """Used to display columns differently based on their type."""

    CATEGORICAL = 1
    CATEGORICAL_SIMILARITY = 2
    CONTINUOUS = 3


class UtilityTesting:
    """A universal set of utilities that let you to compare quality of original vs synthetic data."""

    def __init__(self, synthesizer: Synthesizer, df_orig: pd.DataFrame, df_test: pd.DataFrame, df_synth: pd.DataFrame):
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

        self.vf = synthesizer.value_factory

        # Set the style of plots
        plt.style.use('seaborn')
        mpl.rcParams["axes.facecolor"] = 'w'
        mpl.rcParams['grid.color'] = 'grey'
        mpl.rcParams['grid.alpha'] = 0.1

        mpl.rcParams['axes.linewidth'] = 0.3
        mpl.rcParams['axes.edgecolor'] = 'grey'

        mpl.rcParams['axes.spines.right'] = True
        mpl.rcParams['axes.spines.top'] = True

    def show_distributions(self, remove_outliers: float = 0.0, figsize: Tuple[float, float] = None,
                           cols: int = 2, sample_size: int = 10_000) -> None:
        """Plot comparison plots of all variables in the original and synthetic datasets.

        Args:
            remove_outliers: Percent of outliers to remove.
            figsize: width, height in inches.
            cols: Number of columns in the plot grid.
        """
        if not figsize:
            figsize = (14, 5 * len(self.display_types) + 2)

        fig = plt.figure(figsize=figsize)
        for i, (col, dtype) in enumerate(self.display_types.items()):
            ax = fig.add_subplot(len(self.display_types), cols, i + 1)
            title = col

            col_test = self.df_orig[col].dropna()
            col_synth = self.df_synth[col].dropna()
            if len(col_test) == 0 or len(col_synth) == 0:
                continue

            if dtype == DisplayType.CATEGORICAL:

                df_col_test = pd.DataFrame(col_test)
                df_col_synth = pd.DataFrame(col_synth)

                # We sample orig and synth them so that they have the same size to make the plots more comprehensive
                sample_size = min(sample_size, len(col_test), len(col_synth))
                concatenated = pd.concat([df_col_test.assign(dataset='orig').sample(sample_size),
                                          df_col_synth.assign(dataset='synth').sample(sample_size)])

                ax = sns.countplot(x=col, hue='dataset', data=concatenated,
                                   palette={'orig': COLOR_ORIG, 'synth': COLOR_SYNTH}, ax=ax)

                ax.set_xticklabels(ax.get_xticklabels(), rotation=15)

                emd_distance = metrics.earth_movers_distance(self.df_orig, self.df_synth, col, vf=self.vf)
                title += ' (EMD Dist={:.3f})'.format(emd_distance)

            elif dtype == DisplayType.CONTINUOUS:
                col_test = pd.to_numeric(self.df_orig[col].dropna(), errors='coerce').dropna()
                col_synth = pd.to_numeric(self.df_synth[col].dropna(), errors='coerce').dropna()

                col_test = col_test.sample(min(sample_size, len(col_test)))
                col_synth = col_synth.sample(min(sample_size, len(col_synth)))

                if len(col_test) == 0 or len(col_synth) == 0:
                    continue

                percentiles = [remove_outliers * 100. / 2, 100 - remove_outliers * 100. / 2]
                start, end = np.percentile(col_test, percentiles)
                if start == end:
                    start, end = min(col_test), max(col_test)

                # In case the synthesized data has overflown and has much different domain
                col_synth = col_synth[(start <= col_synth) & (col_synth <= end)]

                if len(col_synth) == 0:
                    continue

                # workaround for kde failing on datasets with only one value
                if col_test.nunique() < 2 or col_synth.nunique() < 2:
                    kde = False
                    kde_kws = None
                else:
                    kde = True
                    kde_kws = {'clip': (start, end)}

                try:
                    sns.distplot(col_test, color=COLOR_ORIG, label='orig', kde=kde, kde_kws=kde_kws,
                                 hist_kws={'color': COLOR_ORIG, 'range': [start, end]}, ax=ax)
                    sns.distplot(col_synth, color=COLOR_SYNTH, label='synth', kde=kde, kde_kws=kde_kws,
                                 hist_kws={'color': COLOR_SYNTH, 'range': [start, end]}, ax=ax)
                except Exception as e:
                    print('ERROR :: Column {} cant be shown :: {}'.format(col, e))

                ks_distance = metrics.kolmogorov_smirnov_distance(self.df_orig, self.df_synth, col, vf=self.vf)
                title += ' (KS Dist={:.3f})'.format(ks_distance)

            ax.set_title(title)
            plt.legend()
        plt.suptitle('Distributions')
        plt.tight_layout(pad=1.1)
        plt.show()

    def utility(self, target: str, model: str = 'GradientBoosting') -> float:
        """Compute utility.

        Utility is a score of estimator trained on synthetic data.

        Args:
            target: Response variable
            model: The estimator to use (converted to classifier or regressor).

        Returns: Utility score.
        """

        orig_score, synth_score, metric, task = predictive_modelling_comparison(
            self.df_orig, self.df_synth, model=model, y_label=target,
            x_labels=[col for col in self.df_orig.columns if col != target]
        )

        print(metric, " (orig):", orig_score)
        print(metric, " (synth):", synth_score)

        return synth_score

    def show_first_order_metric_distances(self, metric: ColumnComparison, **kwargs):
        logger.debug(f"Showing distances for first-order metric ({metric.name}).")
        metric_vector = ColumnComparisonVector(metric)

        result = metric_vector(self.df_test, self.df_synth, vf=self.vf, **kwargs).dropna()

        if len(result) == 0:
            return 0., 0.

        df = pd.DataFrame(result).reset_index()

        dist_max = float(np.nanmax(result))
        dist_avg = float(np.nanmean(result))

        print(f"Max {metric.name}:", dist_max)
        print(f"Average {metric.name} distance:", dist_avg)

        plt.figure(figsize=(8, int(len(df) / 2) + 2))
        g = sns.barplot(y='index', x=metric.name, data=df)
        g.set_xlim(0.0, 1.0)
        plt.title(f'{metric.name}s')
        plt.show()

        return dist_max, dist_avg

    def show_second_order_metric_matrices(self, metric: TwoColumnMetric,
                                          figsize: Tuple[float, float] = (15, 11), **kwargs) -> None:
        """Plot two correlations matrices: one for the original data and one for the synthetic one.

        Args:
            metric: the two column metric to show.
            figsize: width, height in inches.
        """
        logger.debug(f"Showing matrices for second-order metric ({metric.name}).")

        def show_second_order_metric_matrix(df, title=None, ax=None):
            sns.set(style='white')

            metric_matrix = TwoColumnMetricMatrix(metric)
            matrix = metric_matrix(df, vf=self.vf, **kwargs)

            for c in matrix.columns:
                if matrix.loc[:, c].isna().all() and matrix.loc[c, :].isna().all():
                    matrix.drop(c, axis=1, inplace=True)
                    matrix.drop(c, axis=0, inplace=True)

            # Generate a mask for the upper triangle
            mask = np.zeros_like(matrix, dtype=np.bool)
            mask[np.triu_indices_from(mask)] = True

            # Generate a custom diverging colormap
            cmap = sns.diverging_palette(220, 10, as_cmap=True)

            # Draw the heatmap with the mask and correct aspect ratio
            hm = sns.heatmap(matrix, mask=mask, cmap=cmap, vmin=-1.0, vmax=1.0, center=0,
                             square=True, linewidths=.5, cbar_kws={'shrink': .5}, ax=ax, annot=True, fmt='.2f')

            if ax:
                ax.set_ylim(ax.get_ylim()[0] + .5, ax.get_ylim()[1] - .5)
            if title:
                hm.set_title(title)

        # Set up the matplotlib figure
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
        plt.title(f"{metric.name} Matrices")

        show_second_order_metric_matrix(self.df_test, title=f'Original {metric.name}', ax=ax1)
        show_second_order_metric_matrix(self.df_synth, title=f'Synthetic {metric.name}', ax=ax2)
        plt.show()

    def show_second_order_metric_distances(self, metric: TwoColumnComparison,
                                           figsize: Tuple[float, float] = None, **kwargs) -> Tuple[float, float]:
        """Plot a barplot with correlation diffs between original anf synthetic columns.

        Args:
            metric: A two column comparison metric
            figsize: width, height in inches.
        """
        logger.debug(f"Showing distances for second-order metric ({metric.name}).")

        metric_matrix = TwoColumnComparisonMatrix(metric)
        distances = np.abs(metric_matrix(self.df_test,  self.df_synth, vf=self.vf, **kwargs))

        result = []
        for i in range(len(distances.index)):
            for j in range(len(distances.columns)):
                if i < j:
                    row_name = distances.index[i]
                    col_name = distances.iloc[:, j].name
                    if pd.notna(distances.iloc[i, j]):
                        result.append({'column': '{} / {}'.format(row_name, col_name), 'distance': distances.iloc[i, j]})

        if not result:
            return 0., 0.

        df = pd.DataFrame.from_records(result)
        if figsize is None:
            figsize = (10, len(df) // 6 + 2)

        corr_dist_max = float(np.nanmax(df['distance']))
        corr_dist_avg = float(np.nanmean(df['distance']))

        print(f"Max {metric.name}:", corr_dist_max)
        print(f"Average {metric.name}:", corr_dist_avg)

        plt.figure(figsize=figsize)
        plt.title(metric.name)
        g = sns.barplot(y='column', x='distance', data=df)
        g.set_xlim(0.0, 1.0)
        plt.show()

        return corr_dist_max, corr_dist_avg

    def metric_mean_max(self, metric, **kwargs):
        if isinstance(metric, ColumnComparison):
            metric = ColumnComparisonVector(metric)
        elif isinstance(metric, TwoColumnComparison):
            metric = TwoColumnComparisonMatrix(metric)

        x = metric(self.df_orig, self.df_synth, vf=self.vf, **kwargs)

        if len(x) > 0:
            x = x.values
            return float(np.nanmean(x)), float(np.nanmax(x))
        else:
            return 0., 0.

    @staticmethod
    def show_results(current_result, ax=None):
        g = sns.barplot(x=list(current_result.keys()), y=list(current_result.values()), ax=ax, palette='Paired')

        values = list(current_result.values())
        for i in range(len(values)):
            v = values[i]
            g.text(i, v, round(v, 3), color='black', ha="center")

        if ax:
            for tick in ax.get_xticklabels():
                tick.set_rotation(10)
        else:
            plt.xticks(rotation=10)

        plt.show()
