import logging
import math
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes, SubplotBase
from matplotlib.colors import SymLogNorm

from .style import COLOR_ORIG, COLOR_SYNTH, set_plotting_style
from ...insight.metrics import KolmogorovSmirnovDistance
from ...metadata.factory import MetaExtractor
from ...model import DataFrameModel
from ...model.base import ContinuousModel, DiscreteModel
from ...model.factory import ModelFactory

logger = logging.getLogger(__name__)
kolmogorov_smirnov_distance = KolmogorovSmirnovDistance()


def plot_cross_table(counts: pd.DataFrame, title: str, ax=None):
    # Generate a mask for the upper triangle
    sns.set(style='white')
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    hm = sns.heatmap(
        counts, cmap=cmap,
        norm=SymLogNorm(linthresh=.1, vmin=-np.max(counts.values), vmax=np.max(counts.values), clip=True),
        square=True, linewidths=.5, cbar=False, ax=ax, annot=True, fmt='d'
    )

    if ax:
        ax.set_ylim(ax.get_ylim()[0] + .5, ax.get_ylim()[1] - .5)
    if title:
        hm.set_title(title)


def plot_cross_tables(
        df_test: pd.DataFrame, df_synth: pd.DataFrame, col_a: str, col_b: str,
        figsize: Tuple[float, float] = (15, 11)
):
    categories_a = pd.concat((df_test[col_a], df_synth[col_a])).unique()
    categories_b = pd.concat((df_test[col_b], df_synth[col_b])).unique()
    categories_a.sort()
    categories_b.sort()

    # Set up the matplotlib figure
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    plt.title(f"{col_a}:{col_b} Cross Table")
    plot_cross_table(pd.crosstab(
        pd.Categorical(df_test[col_a], categories_a, ordered=True),
        pd.Categorical(df_test[col_b], categories_b, ordered=True),
        dropna=False
    ), title='Original', ax=ax1)
    plot_cross_table(pd.crosstab(
        pd.Categorical(df_synth[col_a], categories_a, ordered=True),
        pd.Categorical(df_synth[col_b], categories_b, ordered=True),
        dropna=False
    ), title='Synthetic', ax=ax2)
    for ax in [ax1, ax2]:
        ax.set_xlabel(col_b)
        ax.set_ylabel(col_a)


def categorical_distribution_plot(col_test, col_synth, sample_size=10_000, ax: Union[Axes, SubplotBase] = None):
    col_test = col_test.dropna()
    col_synth = col_synth.dropna()
    set_plotting_style()
    if len(col_test) == 0 or len(col_synth) == 0:
        return
    df_col_test = pd.DataFrame(col_test)
    df_col_synth = pd.DataFrame(col_synth)
    # We sample orig and synth them so that they have the same size to make the plots more comprehensive
    sample_size = min(sample_size, len(col_test), len(col_synth))
    concatenated = pd.concat([df_col_test.assign(dataset='orig').sample(sample_size),
                              df_col_synth.assign(dataset='synth').sample(sample_size)])
    ax = sns.countplot(x=col_test.name, hue='dataset', data=concatenated, lw=1, alpha=0.7,
                       palette={'orig': COLOR_ORIG, 'synth': COLOR_SYNTH}, ax=ax)
    concatenated["dataset"] = concatenated["dataset"].apply(lambda x: '_' + x)
    ax = sns.countplot(x=col_test.name, hue='dataset', data=concatenated, lw=1,
                       palette={'_orig': COLOR_ORIG, '_synth': COLOR_SYNTH}, fill=False, ax=ax)
    tick_labels = ax.get_xticklabels()
    for tl in tick_labels:
        tl.set_text(tl.get_text()[:25])  # some labels are too long to show completely
    ax.set_xticklabels(tick_labels, rotation=15, ha='right')
    ax.tick_params('y', length=3, width=1, which='major', color='#D7E0FE')
    plt.legend()


def continuous_distribution_plot(col_test, col_synth, remove_outliers: float = 0.0, sample_size=10_000,
                                 ax: Union[Axes, SubplotBase] = None):
    ax = ax or plt.gca()
    set_plotting_style()
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

    col_test = col_test[(start <= col_test) & (col_test <= end)]

    # In case the synthesized data has overflown and has much different domain
    col_synth = col_synth[(start <= col_synth) & (col_synth <= end)]
    if len(col_synth) == 0:
        return
    # workaround for kde failing on datasets with only one value
    if col_test.nunique() < 2 or col_synth.nunique() < 2:
        kde_kws = {}
    else:
        kde_kws = {'clip': (start, end)}
    try:
        sns.kdeplot(col_test, lw=1, alpha=0.5, shade=True, color=COLOR_ORIG, label='orig', ax=ax, **kde_kws)
        sns.kdeplot(col_synth, lw=1, alpha=0.5, shade=True, color=COLOR_SYNTH, label='synth', ax=ax, **kde_kws)
    except Exception as e:
        logger.error('Column {} cant be shown :: {}'.format(col_test.name, e))
    ax.tick_params('y', length=3, width=1, which='major', color='#D7E0FE')
    plt.legend()


def show_distributions(df_orig: pd.DataFrame, df_synth: pd.DataFrame, df_model: Optional[DataFrameModel] = None,
                       remove_outliers: float = 0.0, figsize: Tuple[float, float] = None,
                       cols: int = 2, sample_size: int = 10_000) -> None:
    """Plot comparison plots of all variables in the original and synthetic datasets.

    Args:
        remove_outliers: Percent of outliers to remove.
        figsize: width, height in inches.
        cols: Number of columns in the plot grid.
        sample_size: Maximum sample size tot show distributions.
    """
    if df_model is None:
        df_meta = MetaExtractor.extract(df_orig)
        df_model = ModelFactory()(df_meta)

    rows = math.ceil(len(df_model) / cols)
    if not figsize:
        figsize = (6 * cols + 2, 5 * rows + 2)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=rows, ncols=cols, left=.05, bottom=.05, right=.95, top=.95, wspace=.2, hspace=.3)
    n = 0

    for col, model in df_model.items():

        if model.meta.dtype == 'M8[ns]':
            # Make sure we skip dates when plotting
            continue

        if isinstance(model, DiscreteModel):
            ax = fig.add_subplot(gs[n // cols, n % cols])
            ax.set_title(col)
            categorical_distribution_plot(df_orig[col], df_synth[col], sample_size, ax=ax)
            ax.get_legend().remove()
            ax.set_xlabel("")
            n += 1

        elif isinstance(model, ContinuousModel):
            ax = fig.add_subplot(gs[n // cols, n % cols])
            ax.set_title(col)
            continuous_distribution_plot(df_orig[col], df_synth[col], remove_outliers, sample_size, ax)
            ax.get_legend().remove()
            ax.set_xlabel("")
            n += 1

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', prop={'size': 14})
