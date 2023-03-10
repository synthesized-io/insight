import inspect
import logging
import math
import os
import warnings
from typing import Any, Dict, List, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cycler
from matplotlib.axes import Axes, SubplotBase
from matplotlib.colors import SymLogNorm
from matplotlib.figure import Figure

import insight
from insight.check import ColumnCheck

COLOR_ORIG = "#3af163"
COLOR_SYNTH = "#1457fe"
DEFAULT_FIGSIZE = (10, 10)

logger = logging.getLogger(__name__)


def set_plotting_style():
    """
    Sets the default plotting style for matplotlib.
    """
    plt.style.use("seaborn")

    font_file = "SourceSansPro-Regular.ttf"
    try:
        mpl.font_manager.fontManager.addfont(
            os.path.join(os.path.dirname(inspect.getfile(insight)), "fonts", font_file)
        )
        mpl.rc("font", family="Source Sans Pro")
    except FileNotFoundError:
        warnings.warn(f"Unable to load '{font_file}'")

    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["text.color"] = "333333"
    mpl.rcParams["font.family"] = "Source Sans Pro"
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
    """
    Obtains the figure that is associated with the passed in Axes objects. If not ax object is specified, generates a
    new figure along with a new axe object. Returns the figure and the ax as a tuple.
    Args:
        ax: the ax object from which to obtain the figure.
        figsize: the size of the figure to generate if ax is None.
    return: (fig, ax)
    """
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
      Subdivides the given axes into a grid len(col_titles) by len(row_titles) and labels each section.
      Args:
          col_titles: Title for each column.
          row_titles: Title for each row.
          ax: The axes to subdivide.
          sharey: If all the figures should share the same y-axis.
          wspace: Horizontal space between the figures.
          hspace: Vertical space between the figures.
          height_ratios: A list that details the relative height of each section.
          width_ratios: A list that details the relative height of each section.

        Returns:
            List of the newly generated axes.
      """
    cols = len(col_titles)
    rows = len(row_titles)
    assert cols > 0 and rows > 0

    fig, ax = obtain_figure(ax)

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
    """
    Adjusts the tick labels of the ax to make sure they fit in the figure. Also, slightly tilts the docstrings for
    aesthetic purposes.
    Args:
          ax: the ax on which to adjust the tick labels.
    """
    tick_labels = ax.get_xticklabels()
    for tl in tick_labels:
        tl.set_text(tl.get_text()[:25])  # some labels are too long to show completely
    ax.set_xticklabels(tick_labels, rotation=15, ha="right")
    ax.tick_params("y", length=3, width=1, which="major", color="#D7E0FE")


def text_only(text: str, ax: Union[Axes, SubplotBase] = None) -> Figure:
    """
    Plots the given text over the provided ax. If no ax is provided, generates a new ax.
    Args:
        text: the text to display.
        ax: the ax on which to display the text.

    Returns:
        the figure that relates to the ax.
    """
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


def cross_table(counts: pd.DataFrame, title: str, ax: Axes = None) -> Figure:
    """
    Plots a cross table of the provided dataframe of counts.
    Args:
        counts: a dataframe containing the counts of the categories which to plot.
        title: the title of the plot.
        ax: the ax on which to plot the table.

    Returns:
        a figure with the cross table on it.
    """
    set_plotting_style()
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


def cross_tables(
        df_test: pd.DataFrame,
        df_synth: pd.DataFrame,
        col_a: str,
        col_b: str,
        figsize: Tuple[float, float] = (15, 11),
) -> Figure:
    """
         Plots two cross tables for two datasets for easier comparison of the datasets.
    Args:
        df_test: The first dataframe to plot.
        df_synth: The second dataframe to plot.
        col_a: Categorical column name present in both dataframes.
        col_b: Categorical column name present in both dataframes.
        figsize: Size of the figure.

    Returns:
        A figure with the crosstable on it.
    """
    set_plotting_style()
    categories_a = pd.concat((df_test[col_a], df_synth[col_a])).unique()
    categories_b = pd.concat((df_test[col_b], df_synth[col_b])).unique()
    categories_a.sort()
    categories_b.sort()

    # Set up the matplotlib figure
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    plt.title(f"{col_a}:{col_b} Cross Table")
    cross_table(
        pd.crosstab(
            pd.Categorical(df_test[col_a], categories_a, ordered=True),
            pd.Categorical(df_test[col_b], categories_b, ordered=True),
            dropna=False,
        ),
        title="Original",
        ax=ax1,
    )
    cross_table(
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


def _get_color_scheme(color_scheme, num_cols) -> List[str]:
    """
    Completes the color scheme based on the current matplotlib style. If the color scheme has enough colors to draw the
    specified number of columns, returns the color_scheme unmodified.
    Args:
        color_scheme: the color scheme
        num_cols: number of columns that the color scheme will represent.
    Returns:
        list of string that represent colors.
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if color_scheme is None:
        color_scheme = [colors[i % len(colors)] for i in range(num_cols)]
    elif len(color_scheme) < num_cols:
        color_scheme += [colors[i % len(colors)] for i in range(num_cols - len(color_scheme))]
    return color_scheme


def _get_series_names(source_names, num_cols) -> List[str]:
    """
    Gives names to the unnamed columns. If source names is less than the number of columns, names each column as
    unnamed with an index.
    Args:
        source_names: the original names
        num_cols: total number of columns that need names.

    Returns:
        A list of strings that represents the names of the serieses.
    """
    if source_names is None:
        source_names = ["orig", "synth"] + [f'unnamed{i}' for i in range(num_cols - 2)]
    elif len(source_names) < num_cols:
        source_names += [f'unnamed{i}' for i in range(num_cols - 2)]
    return source_names


def categorical(
        cols: List[pd.Series],
        sample_size=10_000,
        ax: Union[Axes, SubplotBase] = None,
        color_scheme: List[str] = None,
        series_source_names: List[str] = None
) -> Figure:
    """
    Plots the categorical distribution of the specified series side by side.
    Args:
        cols: columns to plot.
        sample_size: maximum number of samples to take from each series in order to make the plots more comperhensive.
            This will need to be larger for larger datasets.
        ax: the ax on which to plot the distributions.
        color_scheme: the colors to use when plotting each column.
        series_source_names: names for each series.

    Returns:
        a figure containing the plot.
    """
    set_plotting_style()
    fig, ax = obtain_figure(ax)
    color_scheme = _get_color_scheme(color_scheme, len(cols))
    series_source_names = _get_series_names(series_source_names, len(cols))
    color_palette = dict(zip(series_source_names, color_scheme))

    df_cols = []
    for i, col in enumerate(cols):
        cols[i] = col.dropna()
        if len(cols[i]) == 0:
            return text_only(f'Column at index {i} is either empty of full of Na\'s')
        df_cols.append(pd.DataFrame(cols[i]))

    sample_size = min(sample_size, min([len(col) for col in cols]))

    # We sample all the columns so that they have the same size to make the plots more comprehensive
    for i, df in enumerate(df_cols):
        df_cols[i] = df.assign(dataset=series_source_names[i]).sample(sample_size)

    concatenated = pd.concat(df_cols)

    ax = sns.countplot(
        x=cols[0].name,
        hue="dataset",
        data=concatenated,
        lw=1,
        alpha=0.7,
        palette=color_palette,
        ax=ax,
        ec='#ffffff'
    )

    adjust_tick_labels(ax)
    if len(cols) <= 1:
        ax.get_legend().remove()

    return fig


def continuous_column(
    col: pd.Series,
    col_name: str = None,
    color: str = None,
    kde_kws: Dict[Any, Any] = None,
    ax: Axes = None
) -> Figure:
    """Plots a pdf of the given continuous series.

    Args:
        col: the series which to plot.
        col_name: the name of the series.
        color: color which is used to plot the series.
        kde_kws: additional arguments for seaborn's kde_plot.
        ax: the ax on which to plot.

    Returns:
        A figure with a pdf of the series plotted on top of it.
    """
    kde_kws = {} if kde_kws is None else kde_kws
    col_name = col.name if col_name is None else col_name

    fig, ax = obtain_figure(ax)
    try:
        if color is None:
            sns.kdeplot(
                col,
                lw=1,
                alpha=0.5,
                fill=True,
                label=col_name,
                ax=ax,
                **kde_kws,
            )
        else:
            sns.kdeplot(
                col,
                lw=1,
                alpha=0.5,
                fill=True,
                color=color,
                label=col_name,
                ax=ax,
                **kde_kws,
            )
    except Exception as e:
        logger.error("Column {} cant be shown :: {}".format(col.name, e))
    ax.tick_params("y", length=3, width=1, which="major", color="#D7E0FE")
    return fig


def continuous(
    cols: List[pd.Series],
    remove_outliers: float = 0.0,
    sample_size=10_000,
    ax: Union[Axes, SubplotBase] = None,
    color_scheme: List[str] = None,
    series_source_names: List[str] = None
) -> Figure:
    """
    plot the pdfs of all the serieses specified by cols.
    Args:
        cols: A list of all the serieses of which to plot.
        remove_outliers: the threshhold for outlier removal.
        sample_size: maximum number of samples to take from each series in order to make the plots more comperhensive.
            This will need to be larger for larger datasets.
        ax: the Axes object on which to plot.
        color_scheme: a list of the colors to use for the plot.
        series_source_names: the names of each series.

    Returns:
        a figure object with the pdfs of each series plotted on it.
    """
    set_plotting_style()
    fig, ax = obtain_figure(ax)
    assert len(cols) > 0
    color_scheme = _get_color_scheme(color_scheme, len(cols))
    series_source_names = _get_series_names(series_source_names, len(cols))

    percentiles = [remove_outliers * 100.0 / 2, 100 - remove_outliers * 100.0 / 2]

    for i, col in enumerate(cols):
        cols[i] = pd.to_numeric(col.dropna(), errors='coerce').dropna()
        if len(cols[i]) == 0:
            return text_only(f'Column at index {i} is empty of full of Na\'s.')
        cols[i] = cols[i].sample(min(sample_size, len(cols[i])))

    start, end = np.percentile(cols[0], percentiles)
    if start == end:
        start, end = min(cols[0]), max(cols[0])

    for i, col in enumerate(cols):
        cols[i] = col[(start <= col) & (col <= end)]
        if len(cols[i]) == 0:
            return text_only(f'Column at index {i} is out of range of the first column.')

    if not all([col.nunique() >= 2 for col in cols]):
        kde_kws = {}
    else:
        kde_kws = {"clip": (start, end)}

    for i, col in enumerate(cols):
        color = color_scheme[i]
        col_name = series_source_names[i]
        continuous_column(col, col_name, color, kde_kws, ax)

    if len(cols) > 1:
        ax.legend()

    return fig


def dataset(
    dfs: List[pd.DataFrame],
    columns=None,
    remove_outliers: float = 0.0,
    figsize: Tuple[float, float] = None,
    figure_cols: int = 2,
    sample_size: int = 10_000,
    max_categories: int = 10,
    check=ColumnCheck()
) -> Figure:
    """
    Plot the columns of all the data-frames that are passed in.
    Args:
        dfs: the dataframes to plot.
        columns: specific columns to plot.
        remove_outliers: the threshold for outlier removal.
        figsize: size of the figure.
        figure_cols: number of columns in the resulting figure.
        sample_size: maximum number of samples to take from each series in order to make the plots more comperhensive.
            This will need to be larger for larger datasets.
        max_categories: maximum number of categories to include for categorical plots. If the number of categories in a
        series exceeds this number, the graph for this series will be replaced by a text.
        check: the check function to use to distinguish between categorical and continuous data.

    Returns:
        a figure containing all the plots for the specified datasets.
    """
    set_plotting_style()
    columns = dfs[0].columns if columns is None else columns

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
        serieses = [dfs[j][col] for j in range(len(dfs))]
        if check.categorical(serieses[0]):
            if pd.concat(serieses).nunique() <= max_categories:
                categorical(serieses, ax=new_ax, sample_size=sample_size)
            else:
                text_only("Number of categories exceeded threshold.", new_ax)
        else:
            continuous(
                serieses, ax=new_ax, remove_outliers=remove_outliers, sample_size=sample_size
            )

        legend = new_ax.get_legend()
        if legend is not None:
            legend.remove()

    if new_ax is not None:
        handles, labels = new_ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", prop={"size": 14})
    return fig
