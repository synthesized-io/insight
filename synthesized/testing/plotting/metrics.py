import logging
from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes, SubplotBase

from .style import COLOR_ORIG, COLOR_SYNTH
from ...insight.evaluation import calculate_evaluation_metrics
from ...insight.metrics import EarthMoversDistance, KolmogorovSmirnovDistance
from ...insight.metrics.metrics_base import (ColumnComparisonVector, DiffMetricMatrix, TwoColumnMetric,
                                             TwoColumnMetricMatrix)
from ...metadata import DataFrameMeta
from ...metadata.factory import MetaExtractor
from ...model import DataFrameModel
from ...model.factory import ModelFactory
from ...util import axes_grid

logger = logging.getLogger(__name__)
kolmogorov_smirnov_distance = KolmogorovSmirnovDistance()
earth_movers_distance = EarthMoversDistance()


def plot_first_order_metric_distances(result: pd.Series, metric_name: str):
    if len(result) == 0:
        return

    df = pd.DataFrame(result.dropna()).reset_index()

    plt.figure(figsize=(8, int(len(df) / 2) + 2))
    g = sns.barplot(y='index', x=metric_name, data=df)
    g.set_xlim(0.0, 1.0)
    plt.title(f'{metric_name}s')


def show_first_order_metric_distances(df_orig: pd.DataFrame, df_synth: pd.DataFrame, df_model: DataFrameModel,
                                      metric: TwoColumnMetric):
    if metric.name is None:
        raise ValueError(f"Given metric {metric} has no name.")
    logger.debug(f"Showing distances for first-order metric ({metric.name}).")
    metric_vector = ColumnComparisonVector(metric)

    result = metric_vector(df_orig, df_synth, df_model=df_model)

    if result is None or len(result.dropna()) == 0:
        return 0., 0.

    dist_max = float(np.nanmax(result))
    dist_avg = float(np.nanmean(result))

    plot_first_order_metric_distances(result, metric_name=metric.name)

    return dist_max, dist_avg


def plot_second_order_metric_matrix(matrix: pd.DataFrame, title: str = None,
                                    ax: Union[Axes, SubplotBase] = None, symmetric: bool = True,
                                    divergent=True):
    # Generate a mask for the upper triangle
    if symmetric:
        mask = np.zeros_like(matrix, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
    else:
        mask = None

    sns.set(style='white')
    # Generate a custom diverging colormap

    if divergent:
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        hm = sns.heatmap(matrix, mask=mask, cmap=cmap, vmin=-1.0, vmax=1.0, center=0,
                         square=True, linewidths=.5, cbar=False, ax=ax, annot=True, fmt='.2f')
    else:
        cmap = sns.light_palette(color=COLOR_SYNTH, as_cmap=True)
        hm = sns.heatmap(matrix, mask=mask, cmap=cmap,
                         square=True, linewidths=.5, cbar=False, ax=ax, annot=True, fmt='.2f')

    if ax:
        ax.set_ylim(ax.get_ylim()[0] + .5, ax.get_ylim()[1] - .5)
        ax.label_outer()
    if title:
        hm.set_title(title)


def plot_second_order_metric_matrices(
        matrix_test: pd.DataFrame, matrix_synth: pd.DataFrame,
        metric_name: str = None, symmetric=True
):
    if len(matrix_test.columns) == 0:
        return

    # Set up the matplotlib figure
    scale = 1.0
    label_length = len(max(matrix_test.columns, key=len)) * 0.08
    width, height = (2 * scale * len(matrix_test.columns) + label_length, scale * len(matrix_test) + label_length)
    figsize = (width, height)

    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    fig.suptitle(f"{metric_name} Matrices", fontweight='bold', fontsize=14)
    ax1, ax2 = axes_grid(ax, rows=1, cols=2, col_titles=['Original', 'Synthetic'], wspace=1 / len(matrix_test.columns))

    left = (0.5 + label_length) / width
    bottom = (0.5 + label_length) / height
    right = 1 - (0.5 / width)
    top = 1 - (1 / height)

    ax.get_gridspec().update(
        left=left,
        bottom=bottom,
        right=right,
        top=top if top > bottom else bottom * 1.1
    )

    plot_second_order_metric_matrix(matrix_test, ax=ax1, symmetric=symmetric)
    plot_second_order_metric_matrix(matrix_synth, ax=ax2, symmetric=symmetric)


def _filtered_metric_matrix(df: pd.DataFrame, df_model: DataFrameModel, metric: TwoColumnMetric) -> pd.DataFrame:
    metric_matrix = TwoColumnMetricMatrix(metric)
    matrix = metric_matrix(df, df_model=df_model)
    if matrix is None:
        raise ValueError(f"Unable to compute {metric_matrix.name}")

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


def show_second_order_metric_matrices(df_orig: pd.DataFrame, df_synth: pd.DataFrame, df_model: DataFrameModel,
                                      metric: TwoColumnMetric) -> None:
    """Plot two correlations matrices: one for the original data and one for the synthetic one.

    Args:
        metric: the two column metric to show.
        figsize: width, height in inches.
    """
    if metric.name is None:
        raise ValueError(f"Given metric {metric} has no name.")

    logger.debug(f"Showing matrices for second-order metric ({metric.name}).")

    matrix_orig = _filtered_metric_matrix(df_orig, df_model, metric)
    matrix_synth = _filtered_metric_matrix(df_synth, df_model, metric)

    is_symmetric = True if 'symmetric' in metric.tags else False

    plot_second_order_metric_matrices(matrix_orig, matrix_synth, metric.name, symmetric=is_symmetric)


def plot_second_order_metric_distances(df: pd.DataFrame, metric_name: str, figsize=None):
    if figsize is None:
        figsize = (10, len(df) // 6 + 2)
    plt.figure(figsize=figsize)
    plt.title(metric_name)
    g = sns.barplot(y='column', x='distance', data=df)
    g.set_xlim(0.0, 1.0)


def show_second_order_metric_distances(df_orig: pd.DataFrame, df_synth: pd.DataFrame, df_model: DataFrameModel,
                                       metric: TwoColumnMetric) -> Tuple[float, float]:
    """Plot a barplot with correlation diffs between original anf synthetic columns.

    Args:
        metric: A two column comparison metric
    """
    if metric.name is None:
        raise ValueError(f"Given metric {metric} has no name.")

    logger.debug(f"Showing distances for second-order metric ({metric.name}).")

    metric_matrix = TwoColumnMetricMatrix(metric)
    diff_metric_matrix = DiffMetricMatrix(metric_matrix)

    distances = np.abs(diff_metric_matrix(df_orig, df_synth, df_model=df_model))

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

    plot_second_order_metric_distances(df, metric.name)

    return corr_dist_max, corr_dist_avg


def plot_standard_metrics(df_test: pd.DataFrame, df_synth: pd.DataFrame, df_meta: DataFrameMeta = None,
                          ax: plt.Axes = None, sample_size: int = None) -> Dict[str, float]:

    if sample_size is not None:
        if sample_size < len(df_test):
            df_test = df_test.sample(sample_size)
        if sample_size < len(df_synth):
            df_synth = df_synth.sample(sample_size)

    if df_meta is None:
        df_meta = MetaExtractor.extract(pd.concat((df_test, df_synth)))

    df_model = ModelFactory()(df_meta)
    standard_metrics = calculate_evaluation_metrics(df_test, df_synth, df_model=df_model)

    current_result = dict()
    for name, val in standard_metrics.items():
        if len(val) > 0 and not np.all(val.isna()):
            x = val.to_numpy()
            x_avg, x_max = float(np.nanmean(x)), float(np.nanmax(x))
        else:
            x_avg, x_max = 0., 0.

        current_result[f'{name}_avg'] = x_avg
        current_result[f'{name}_max'] = x_max

    bar_plot_results(current_result, ax=ax)

    return current_result


def bar_plot_results(current_result, ax: Union[Axes, SubplotBase] = None):
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


def plt_dist_orig_snyth(df_orig: pd.DataFrame, df_synth: pd.DataFrame, key: str, unique_threshold: int = 30,
                        ax: plt.Axes = None, sample_size: int = 10_000) -> float:

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    orig_vc = pd.DataFrame(df_orig[key].value_counts())
    synth_vc = pd.DataFrame(df_synth[key].value_counts())

    if orig_vc[key].nunique(dropna=False) > unique_threshold:
        #     if True:
        all_vc = np.concatenate((orig_vc[key].values, synth_vc[key].values))
        remove_outliers = 0.02
        percentiles = [remove_outliers * 100. / 2, 100 - remove_outliers * 100. / 2]
        start, end = np.percentile(all_vc, percentiles)
        if start == end:
            start, end = min(all_vc), max(all_vc)

        orig_vc_values = orig_vc.loc[(start <= orig_vc[key]) & (orig_vc[key] <= end), key].values
        synth_vc_values = synth_vc.loc[(start <= synth_vc[key]) & (synth_vc[key] <= end), key].values

        sns.distplot(orig_vc_values, color=COLOR_ORIG, label='orig', ax=ax)
        sns.distplot(synth_vc_values, color=COLOR_SYNTH, label='synth', ax=ax)
        dist = kolmogorov_smirnov_distance(orig_vc[key], synth_vc[key])

    else:
        # We sample orig and synth them so that they have the same size to make the plots more comprehensive
        sample_size = min(len(orig_vc), len(synth_vc), sample_size)
        concatenated = pd.concat([orig_vc.assign(dataset='orig').sample(sample_size),
                                  synth_vc.assign(dataset='synth').sample(sample_size)])

        ax = sns.countplot(x=key, hue='dataset', data=concatenated,
                           palette={'orig': COLOR_ORIG, 'synth': COLOR_SYNTH}, ax=ax)
        dist = earth_movers_distance(orig_vc[key], synth_vc[key])

    ax.legend()
    return float(dist or 0.0)
