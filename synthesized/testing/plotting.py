import logging
import time
from typing import Optional, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.axes import Axes

from ..insight.metrics import kolmogorov_smirnov_distance


logger = logging.getLogger(__name__)

MAX_SAMPLE_DATES = 2500
NUM_UNIQUE_CATEGORICAL = 100
NAN_FRACTION_THRESHOLD = 0.25
NON_NAN_COUNT_THRESHOLD = 500
CATEGORICAL_THRESHOLD_LOG_MULTIPLIER = 2.5

COLOR_ORIG = '#00AB26'
COLOR_SYNTH = '#2794F3'


# -- Plotting functions
def set_plotting_style():
    plt.style.use('seaborn')
    mpl.rcParams["axes.facecolor"] = 'w'
    mpl.rcParams['grid.color'] = 'grey'
    mpl.rcParams['grid.alpha'] = 0.1

    mpl.rcParams['axes.linewidth'] = 0.3
    mpl.rcParams['axes.edgecolor'] = 'grey'

    mpl.rcParams['axes.spines.right'] = True
    mpl.rcParams['axes.spines.top'] = True


def plot_data(data: pd.DataFrame, ax: Axes):
    """Plot one- or two-dimensional dataframe `data` on `matplotlib` axis `ax` according to column types. """
    if data.shape[1] == 1:
        if data['x'].dtype.kind == 'O':
            return sns.countplot(data["x"], ax=ax)
        else:
            return sns.distplot(data["x"], ax=ax)
    elif data.shape[1] == 2:
        if data['x'].dtype.kind in {'O', 'i'} and data['y'].dtype.kind == 'f':
            sns.violinplot(x="x", y="y", data=data, ax=ax)
        elif data['x'].dtype.kind == 'f' and data['y'].dtype.kind == 'f':
            return ax.hist2d(data['x'], data['y'], bins=100)
        elif data['x'].dtype.kind == 'O' and data['y'].dtype.kind == 'O':
            crosstab = pd.crosstab(data['x'], columns=[data['y']]).apply(lambda r: r/r.sum(), axis=1)
            sns.heatmap(crosstab, vmin=0.0, vmax=1.0, ax=ax)
        else:
            return sns.distplot(data, ax=ax, color=["b", "g"])
    else:
        return sns.distplot(data, ax=ax)


def plot_multidimensional(original: pd.DataFrame, synthetic: pd.DataFrame, ax: Axes = None):
    """
    Plot Kolmogorov-Smirnov distance between the columns in the dataframes
    `original` and `synthetic` on `matplotlib` axis `ax`.
    """
    dtype_dict = {"O": "Categorical", "i": "Categorical", "f": "Continuous"}
    default_palette = sns.color_palette()
    color_dict = {"Categorical": default_palette[0], "Continuous": default_palette[1]}
    assert (original.columns == synthetic.columns).all(), "Original and synthetic data must have the same columns."
    columns = original.columns.values.tolist()
    error_msg = "Original and synthetic data must have the same data types."
    assert (original.dtypes.values == synthetic.dtypes.values).all(), error_msg
    dtypes = [dtype_dict[dtype.kind] for dtype in original.dtypes.values]
    distances = [kolmogorov_smirnov_distance(original, synthetic, col) for col in original.columns]
    plot = sns.barplot(x=columns, y=distances, hue=dtypes, ax=ax, palette=color_dict, dodge=False)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=30)
    plot.set_title("KS distance by column")
    return plot


def plot_first_order_metric_distances(result: pd.Series, metric_name: str):
    if len(result) == 0:
        return

    df = pd.DataFrame(result.dropna()).reset_index()

    plt.figure(figsize=(8, int(len(df) / 2) + 2))
    g = sns.barplot(y='index', x=metric_name, data=df)
    g.set_xlim(0.0, 1.0)
    plt.title(f'{metric_name}s')
    plt.show()


def plot_second_order_metric_matrix(matrix: pd.DataFrame, title: str, ax=None):
    # Generate a mask for the upper triangle
    mask = np.zeros_like(matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    sns.set(style='white')
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    hm = sns.heatmap(matrix, mask=mask, cmap=cmap, vmin=-1.0, vmax=1.0, center=0,
                     square=True, linewidths=.5, cbar_kws={'shrink': .5}, ax=ax, annot=True, fmt='.2f')

    if ax:
        ax.set_ylim(ax.get_ylim()[0] + .5, ax.get_ylim()[1] - .5)
    if title:
        hm.set_title(title)


def plot_second_order_metric_matrices(
        matrix_test: pd.DataFrame, matrix_synth: pd.DataFrame,
        metric_name: str, figsize: Tuple[float, float] = (15, 11)
):
    # Set up the matplotlib figure
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    plt.title(f"{metric_name} Matrices")
    plot_second_order_metric_matrix(matrix_test, title=f'Original {metric_name}', ax=ax1)
    plot_second_order_metric_matrix(matrix_synth, title=f'Synthetic {metric_name}', ax=ax2)
    plt.show()


def plot_second_order_metric_distances(df: pd.DataFrame, metric_name: str, figsize=None):
    if figsize is None:
        figsize = (10, len(df) // 6 + 2)
    plt.figure(figsize=figsize)
    plt.title(metric_name)
    g = sns.barplot(y='column', x='distance', data=df)
    g.set_xlim(0.0, 1.0)
    plt.show()


def bar_plot_results(current_result, ax=None):
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


def categorical_distribution_plot(col_test, col_synth, title, sample_size=10_000, ax=None):
    col_test = col_test.dropna()
    col_synth = col_synth.dropna()

    if len(col_test) == 0 or len(col_synth) == 0:
        return

    df_col_test = pd.DataFrame(col_test)
    df_col_synth = pd.DataFrame(col_synth)

    # We sample orig and synth them so that they have the same size to make the plots more comprehensive
    sample_size = min(sample_size, len(col_test), len(col_synth))
    concatenated = pd.concat([df_col_test.assign(dataset='orig').sample(sample_size),
                              df_col_synth.assign(dataset='synth').sample(sample_size)])

    ax = sns.countplot(x=col_test.name, hue='dataset', data=concatenated,
                       palette={'orig': COLOR_ORIG, 'synth': COLOR_SYNTH}, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
    ax.set_title(title)
    plt.legend()


def continuous_distribution_plot(col_test, col_synth, title, remove_outliers: float = 0.0, sample_size=10_000, ax=None):
    col_test = pd.to_numeric(col_test.dropna(), errors='coerce').dropna()
    col_synth = pd.to_numeric(col_synth.dropna(), errors='coerce').dropna()
    if len(col_test) == 0 or len(col_synth) == 0:
        return

    col_test = col_test.sample(min(sample_size, len(col_test)))
    col_synth = col_synth.sample(min(sample_size, len(col_synth)))

    percentiles = [remove_outliers * 100. / 2, 100 - remove_outliers * 100. / 2]
    start, end = np.percentile(col_test, percentiles)
    if start == end:
        start, end = min(col_test), max(col_test)

    # In case the synthesized data has overflown and has much different domain
    col_synth = col_synth[(start <= col_synth) & (col_synth <= end)]
    if len(col_synth) == 0:
        return

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
        print('ERROR :: Column {} cant be shown :: {}'.format(col_test.name, e))

    ax.set_title(title)
    plt.legend()


def plot_time_series(x, t, ax):
    kind = x.dtype.kind
    if kind in {"i", "f"}:
        sequence_line_plot(x=x, t=t, ax=ax)
    else:
        sequence_index_plot(x=x, t=t, ax=ax)


def plot_auto_association(original: np.array, synthetic: np.array, axes: np.array):
    assert axes is not None
    lags = list(range(original.shape[-1]))
    axes[0].stem(lags, original, "g", markerfmt='go', use_line_collection=True)
    axes[0].set_title("Original")
    axes[1].stem(lags, synthetic, "b", markerfmt='bo', use_line_collection=True)
    axes[1].set_title("Synthetic")


def sequence_index_plot(x, t, ax: Axes, cmap_name: str = "YlGn"):
    values = np.unique(x)
    val2idx = {val: i for i, val in enumerate(values)}
    cmap = cm.get_cmap(cmap_name)
    colors = [cmap(j/values.shape[0]) for j in range(values.shape[0])]

    for i, val in enumerate(x):
        ax.fill_between((i, i+1), 2, facecolor=colors[val2idx[val]])
    ax.get_yaxis().set_visible(False)


def sequence_line_plot(x, t, ax):
    sns.lineplot(x=t, y=x, ax=ax)
