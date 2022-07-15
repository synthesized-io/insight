import logging
import math
import warnings
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pkg_resources
import seaborn as sns
from matplotlib import cycler
from matplotlib.axes import Axes, SubplotBase
from matplotlib.colors import SymLogNorm
from matplotlib.figure import Figure

from synthesized_insight.check import Check, ColumnCheck

COLOR_ORIG = "#FF4D5B"
COLOR_SYNTH = "#312874"
DEFAULT_FIGSIZE = (10, 10)

logger = logging.getLogger(__name__)


def set_plotting_style():
    plt.style.use("seaborn")
    font_file = "fonts/inter-v3-latin-regular.ttf"
    try:
        mpl.font_manager.fontManager.addfont(
            Path(pkg_resources.resource_filename("synthesized_insight", font_file)).as_posix()
        )
        mpl.rc("font", family="Inter-Regular")
    except FileNotFoundError:
        warnings.warn(f"Unable to load '{font_file}'")

    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["text.color"] = "333333"
    mpl.rcParams["font.family"] = "inter"
    mpl.rcParams["axes.facecolor"] = "EFF3FF"
    mpl.rcParams["axes.edgecolor"] = "333333"
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.color"] = "D7E0FE"
    mpl.rcParams["xtick.direction"] = "out"
    mpl.rcParams["ytick.direction"] = "out"
    mpl.rcParams["axes.prop_cycle"] = cycler(
        "color", ["312874", "FF4D5B", "FFBDD1", "4EC7BD", "564E9C"] * 10
    )


def obtain_figure(ax: Axes = None, figsize: Tuple[int, int] = DEFAULT_FIGSIZE):
    set_plotting_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax


def axes_grid(
        col_titles: List[str],
        row_titles: List[str],
        ax: Union[Axes, SubplotBase] = None,
        sharey: bool = True,
        wspace: float = None,
        hspace: float = None,
        height_ratios: float = None,
        width_ratios: float = None,
):
    """
    Args:
        ax: The axes to subdivide.
        col_titles: Title for each column.
        row_titles: Title for each row.
        sharey:
        wspace:
        hspace:
        height_ratios:
        width_ratios:
    """
    cols = len(col_titles)
    rows = len(row_titles)
    assert cols > 0 and rows > 0

    if ax is None:
        ax = plt.figure(figsize=DEFAULT_FIGSIZE).gca()

    ax.set_axis_off()
    sp_spec = ax.get_subplotspec()
    sgs = sp_spec.subgridspec(rows, cols, wspace=wspace, hspace=hspace, height_ratios=height_ratios,
                              width_ratios=width_ratios)
    fig = ax.figure
    col_axes: List[mpl.axes.Axes] = list()

    ax = fig.add_subplot(sgs[:, 0])
    ax.set_title(col_titles[0])
    col_axes.append(ax)
    sy = ax if sharey else None

    for c in range(1, cols):
        ax = fig.add_subplot(sgs[:, c], sharey=sy)
        ax.set_title(col_titles[c])
        col_axes.append(ax)

    if rows == 1:
        col_axes[0].set_ylabel(row_titles[0])

        return col_axes

    for col_ax in col_axes:
        col_ax.set_axis_off()

    axes = []

    if cols == 1:
        axes.append(fig.add_subplot(sgs[0, 0]))
        axes[0].set_ylabel(row_titles[0])

        for r in range(1, rows):
            axes.append(fig.add_subplot(sgs[r, 0], sharex=axes[0]))
            axes[r].set_ylabel(row_titles[r])

        return axes

    row_axes = [fig.add_subplot(sgs[0, 0])]
    row_axes[0].set_ylabel(row_titles[0])
    sy = row_axes[0] if sharey else None

    for c in range(1, cols):
        row_axes.append(fig.add_subplot(sgs[0, c], sharey=sy))
    axes.append(row_axes)

    for r in range(1, rows):
        row_axes = list()
        row_axes.append(fig.add_subplot(sgs[r, 0], sharex=axes[0][0]))
        row_axes[0].set_ylabel(row_titles[r])
        sy = row_axes[0] if sharey else None

        for c in range(1, cols):
            row_axes.append(fig.add_subplot(sgs[r, c], sharex=axes[0][c], sharey=sy))
        axes.append(row_axes)

    return axes


def adjust_tick_labels(ax: Axes):
    tick_labels = ax.get_xticklabels()
    for tl in tick_labels:
        tl.set_text(tl.get_text()[:25])  # some labels are too long to show completely
    ax.set_xticklabels(tick_labels, rotation=15, ha="right")
    ax.tick_params("y", length=3, width=1, which="major", color="#D7E0FE")


def plot_text_only(text: str, ax: Union[Axes, SubplotBase] = None) -> Figure:
    set_plotting_style()
    fig, ax = obtain_figure(ax)
    ax.text(
        0.5,
        0.5,
        text,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    return fig


def plot_cross_table(counts: pd.DataFrame, title: str, ax: Axes = None) -> Figure:
    fig, ax = obtain_figure(ax)
    # Generate a mask for the upper triangle
    sns.set(style="white")
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    hm = sns.heatmap(
        counts,
        cmap=cmap,
        norm=SymLogNorm(
            linthresh=0.1,
            vmin=-np.max(counts.values),
            vmax=np.max(counts.values),
            clip=True,
        ),
        square=True,
        linewidths=0.5,
        cbar=False,
        ax=ax,
        annot=True,
        fmt="d",
    )

    if ax:
        ax.set_ylim(ax.get_ylim()[0] + 0.5, ax.get_ylim()[1] - 0.5)
    if title:
        hm.set_title(title)
    return fig


def plot_cross_tables(
    df_test: pd.DataFrame,
    df_synth: pd.DataFrame,
    col_a: str,
    col_b: str,
    figsize: Tuple[float, float] = (15, 11),
) -> Figure:
    categories_a = pd.concat((df_test[col_a], df_synth[col_a])).unique()
    categories_b = pd.concat((df_test[col_b], df_synth[col_b])).unique()
    categories_a.sort()
    categories_b.sort()

    # Set up the matplotlib figure
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    plt.title(f"{col_a}:{col_b} Cross Table")
    plot_cross_table(
        pd.crosstab(
            pd.Categorical(df_test[col_a], categories_a, ordered=True),
            pd.Categorical(df_test[col_b], categories_b, ordered=True),
            dropna=False,
        ),
        title="Original",
        ax=ax1,
    )
    plot_cross_table(
        pd.crosstab(
            pd.Categorical(df_synth[col_a], categories_a, ordered=True),
            pd.Categorical(df_synth[col_b], categories_b, ordered=True),
            dropna=False,
        ),
        title="Synthetic",
        ax=ax2,
    )
    for ax in [ax1, ax2]:
        ax.set_xlabel(col_b)
        ax.set_ylabel(col_a)

    return f


def categorical_distribution_plot(
        col_test: pd.Series,
        col_synth: pd.Series = None,
        sample_size: int = 10_000,
        ax: Union[Axes, SubplotBase] = None
) -> Figure:
    single_column = True
    fig, ax = obtain_figure(ax)
    col_test = col_test.dropna()

    if col_synth is not None:
        single_column = False
        col_synth = col_synth.dropna()

    if len(col_test) == 0 or (not single_column and len(col_synth) == 0):
        return fig

    df_col_test = pd.DataFrame(col_test)
    df_col_synth = pd.DataFrame(col_synth) if not single_column else None

    if not single_column:
        # We sample orig and synth so that they have the same size to make the plots more comprehensive
        sample_size = min(sample_size, len(col_test), len(col_synth))
        concatenated = pd.concat(
            [
                df_col_test.assign(dataset="orig").sample(sample_size),
                df_col_synth.assign(dataset="synth").sample(sample_size),
            ]
        )
    else:
        # No need to sample since we are only plotting one column.
        concatenated = df_col_test.assign(dataset="orig")

    ax = sns.countplot(
        x=col_test.name,
        hue="dataset",
        data=concatenated,
        lw=1,
        alpha=0.7,
        palette={"orig": COLOR_ORIG, "synth": COLOR_SYNTH},
        ax=ax,
        ec='#ffffff'
    )
    adjust_tick_labels(ax)
    if not single_column:
        plt.legend()

    return fig


def continuous_distribution_plot(
        col_test: pd.Series,
        col_synth: pd.Series = None,
        remove_outliers: float = 0.0,
        sample_size: int = 10_000,
        ax: Union[Axes, SubplotBase] = None,
) -> Figure:
    fig, ax = obtain_figure(ax)
    single_column = True

    col_test = pd.to_numeric(col_test.dropna(), errors="coerce").dropna()

    if col_synth is not None:
        single_column = False
        col_synth = pd.to_numeric(col_synth.dropna(), errors="coerce").dropna()

    if len(col_test) == 0 or (not single_column and len(col_synth) == 0):
        return fig

    col_test = col_test.sample(min(sample_size, len(col_test)))
    col_synth = col_synth.sample(min(sample_size, len(col_synth))) if not single_column else None

    percentiles = [remove_outliers * 100.0 / 2, 100 - remove_outliers * 100.0 / 2]
    start, end = np.percentile(col_test, percentiles)

    if start == end:
        start, end = min(col_test), max(col_test)

    col_test = col_test[(start <= col_test) & (col_test <= end)]

    if not single_column:
        # In case the synthesized data has overflown and has much different domain
        col_synth = col_synth[(start <= col_synth) & (col_synth <= end)]
        if len(col_synth) == 0:
            return fig

    # workaround for kde failing on datasets with only one value
    if col_test.nunique() < 2 or (not single_column and col_synth.nunique() < 2):
        kde_kws = {}
    else:
        kde_kws = {"clip": (start, end)}
    try:
        sns.kdeplot(
            col_test,
            lw=1,
            alpha=0.5,
            shade=True,
            color=COLOR_ORIG,
            label="orig",
            ax=ax,
            **kde_kws,
        )
        if not single_column:
            sns.kdeplot(
                col_synth,
                lw=1,
                alpha=0.5,
                shade=True,
                color=COLOR_SYNTH,
                label="synth",
                ax=ax,
                **kde_kws,
            )
    except Exception as e:
        logger.error("Column {} cant be shown :: {}".format(col_test.name, e))
    ax.tick_params("y", length=3, width=1, which="major", color="#D7E0FE")
    if not single_column:
        plt.legend()
    return fig


def plot_dataset(
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        columns: List[str] = None,
        remove_outliers: float = 0.0,
        figsize: Tuple[float, float] = None,
        figure_cols: int = 2,
        sample_size: int = 10_000,
        max_categories: int = 10,
        check: Check = ColumnCheck()
) -> Figure:
    columns = df_a.columns if columns is None else columns

    figure_rows = math.ceil(len(columns) / figure_cols)
    figsize = (6 * figure_cols + 2, 5 * figure_rows + 2) if figsize is None else figsize
    fig = plt.figure(figsize=figsize)
    fig.suptitle('Distributions')

    if len(columns) == 0:
        return fig

    gs = fig.add_gridspec(
        nrows=figure_rows,
        ncols=figure_cols,
        left=0.05,
        bottom=0.05,
        right=0.95,
        top=0.95,
        wspace=0.2,
        hspace=0.3,
    )

    new_ax = None
    for i, col in enumerate(columns):
        row_pos = i // figure_cols
        col_pos = i % figure_cols
        new_ax = fig.add_subplot(gs[row_pos, col_pos])
        if check.categorical(df_a[col]):
            if pd.concat([df_a[col], df_b[col]]).nunique() <= max_categories:
                categorical_distribution_plot(df_a[col],
                                              df_b[col],
                                              ax=new_ax,
                                              sample_size=sample_size)
            else:
                plot_text_only("Number of categories exceeded threshold.", new_ax)
        else:
            continuous_distribution_plot(df_a[col],
                                         df_b[col],
                                         ax=new_ax,
                                         remove_outliers=remove_outliers,
                                         sample_size=sample_size)

        legend = new_ax.get_legend()
        if legend is not None:
            legend.remove()

    if new_ax is not None:
        handles, labels = new_ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", prop={"size": 14})

    return fig