from enum import Enum
from typing import Tuple
import math
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ..metadata import DataFrameMeta
from .plotting import set_plotting_style, plot_first_order_metric_distances, plot_second_order_metric_distances,\
    plot_second_order_metric_matrices, continuous_distribution_plot, categorical_distribution_plot, plot_standard_metrics
from ..insight import metrics
from ..insight.metrics import TwoColumnComparison, TwoColumnComparisonMatrix
from ..insight.metrics import TwoColumnMetric, TwoColumnMetricMatrix
from ..insight.metrics import ColumnComparison, ColumnComparisonVector
from ..insight.metrics.modelling_metrics import predictive_modelling_comparison

logger = logging.getLogger(__name__)

MAX_PVAL = 0.05


class DisplayType(Enum):
    """Used to display columns differently based on their type."""

    CATEGORICAL = 1
    CATEGORICAL_SIMILARITY = 2
    CONTINUOUS = 3


class Assessor:
    """A universal set of utilities that let you to compare quality of original vs synthetic data."""

    def __init__(self, df_meta: DataFrameMeta, df_orig: pd.DataFrame, df_synth: pd.DataFrame):
        """Create an instance of UtilityTesting.

        Args:
            synthesizer: A synthesizer instance.
            df_orig: A DataFrame with original data that was used for training
            df_test: A DataFrame with hold-out original data
            df_synth: A DataFrame with synthetic data
        """
        self.df_orig = df_orig
        self.df_synth = df_synth

        self.df_meta = df_meta
        categorical, continuous = self.df_meta.get_categorical_and_continuous()

        self.categorical, self.continuous = [v.name for v in categorical], [v.name for v in continuous]
        self.plotable_values = self.categorical + self.continuous

        # Set the style of plots
        set_plotting_style()

    def show_standard_metrics(self, ax=None):
        current_result = plot_standard_metrics(self.df_test, self.df_synth, self.df_meta, ax=ax)
        plt.show()

        return current_result

    def show_distributions(self, remove_outliers: float = 0.0, figsize: Tuple[float, float] = None,
                           cols: int = 2, sample_size: int = 10_000) -> None:
        """Plot comparison plots of all variables in the original and synthetic datasets.

        Args:
            remove_outliers: Percent of outliers to remove.
            figsize: width, height in inches.
            cols: Number of columns in the plot grid.
            sample_size: Maximum sample size tot show distributions.
        """
        rows = math.ceil(len(self.plotable_values) / cols)
        if not figsize:
            figsize = (14, 5 * rows + 2)

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(nrows=rows, ncols=cols, left=.05, bottom=.05, right=.95, top=.95, wspace=.2, hspace=.3)
        n = 0

        for i, col in enumerate(self.categorical):
            ax = fig.add_subplot(gs[n // cols, n % cols])

            emd_distance = metrics.earth_movers_distance(self.df_orig, self.df_synth, col, dp=self.df_meta)
            title = f'{col} (EMD Dist={emd_distance:.3f})'
            categorical_distribution_plot(self.df_orig[col], self.df_synth[col], title, sample_size, ax=ax)
            n += 1

        for i, col in enumerate(self.continuous):
            ax = fig.add_subplot(gs[n // 2, n % 2])

            ks_distance = metrics.kolmogorov_smirnov_distance(self.df_orig, self.df_synth, col, dp=self.df_meta)
            title = f'{col} (KS Dist={ks_distance:.3f})'
            continuous_distribution_plot(self.df_orig[col], self.df_synth[col], title, remove_outliers, sample_size, ax)
            n += 1

        plt.suptitle('Distributions', x=0.5, y=0.98, fontweight='bold')
        plt.show()

    def show_first_order_metric_distances(self, metric: ColumnComparison, **kwargs):
        if metric.name is None:
            raise ValueError("Metric has no name.")
        logger.debug(f"Showing distances for first-order metric ({metric.name}).")
        metric_vector = ColumnComparisonVector(metric)

        result = metric_vector(self.df_orig, self.df_synth, dp=self.df_meta, **kwargs)

        if result is None or len(result.dropna()) == 0:
            return 0., 0.

        dist_max = float(np.nanmax(result))
        dist_avg = float(np.nanmean(result))

        print(f"Max {metric.name}:", dist_max)
        print(f"Average {metric.name} distance:", dist_avg)

        plot_first_order_metric_distances(result, metric_name=metric.name)

        return dist_max, dist_avg

    def show_second_order_metric_matrices(self, metric: TwoColumnMetric, **kwargs) -> None:
        """Plot two correlations matrices: one for the original data and one for the synthetic one.

        Args:
            metric: the two column metric to show.
            figsize: width, height in inches.
        """
        if metric.name is None:
            raise ValueError("Metric has no name.")

        logger.debug(f"Showing matrices for second-order metric ({metric.name}).")

        def filtered_metric_matrix(df):
            metric_matrix = TwoColumnMetricMatrix(metric)
            matrix = metric_matrix(df, dp=self.df_meta, **kwargs)

            for c in matrix.columns:
                if matrix.loc[:, c].isna().all() and matrix.loc[c, :].isna().all():
                    matrix.drop(c, axis=1, inplace=True)
                    matrix.drop(c, axis=0, inplace=True)

            if [c for c in matrix.columns if matrix.loc[:, c].isna().all()] == \
                    [c for c in matrix.columns if not matrix.loc[c, :].isna().all()]:
                for c in matrix.columns:
                    if matrix.loc[:, c].isna().all():
                        matrix.drop(c, axis=1, inplace=True)
                    elif matrix.loc[c, :].isna().all():
                        matrix.drop(c, axis=0, inplace=True)

            return matrix

        matrix_orig = filtered_metric_matrix(self.df_orig)
        matrix_synth = filtered_metric_matrix(self.df_synth)

        is_symmetric = True if 'symmetric' in metric.tags else False

        plot_second_order_metric_matrices(matrix_orig, matrix_synth, metric.name, symmetric=is_symmetric)

    def show_second_order_metric_distances(self, metric: TwoColumnComparison, **kwargs) -> Tuple[float, float]:
        """Plot a barplot with correlation diffs between original anf synthetic columns.

        Args:
            metric: A two column comparison metric
        """
        if metric.name is None:
            raise ValueError("Metric has no name.")

        logger.debug(f"Showing distances for second-order metric ({metric.name}).")

        metric_matrix = TwoColumnComparisonMatrix(metric)
        distances = np.abs(metric_matrix(self.df_orig, self.df_synth, dp=self.df_meta, **kwargs))

        result = []
        for i in range(len(distances.index)):
            for j in range(len(distances.columns)):
                if i < j:
                    row_name = distances.index[i]
                    col_name = distances.iloc[:, j].name
                    if pd.notna(distances.iloc[i, j]):
                        result.append({'column': '{} / {}'.format(row_name, col_name),
                                       'distance': distances.iloc[i, j]})

        if not result:
            return 0., 0.

        df = pd.DataFrame.from_records(result)

        corr_dist_max = float(np.nanmax(df['distance']))
        corr_dist_avg = float(np.nanmean(df['distance']))

        print(f"Max {metric.name}:", corr_dist_max)
        print(f"Average {metric.name}:", corr_dist_avg)

        plot_second_order_metric_distances(df, metric.name)

        return corr_dist_max, corr_dist_avg

    def metric_mean_max(self, metric, **kwargs):
        if isinstance(metric, ColumnComparison):
            metric = ColumnComparisonVector(metric)
        elif isinstance(metric, TwoColumnComparison):
            metric = TwoColumnComparisonMatrix(metric)

        x = metric(self.df_orig, self.df_synth, dp=self.df_meta, **kwargs)

        if len(x) > 0:
            x = x.values
            return float(np.nanmean(x)), float(np.nanmax(x))
        else:
            return 0., 0.
