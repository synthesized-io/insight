import logging
import math
from typing import List, Tuple, Union

import matplotlib as mpl
from matplotlib.colors import SymLogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from matplotlib import cm
from matplotlib.axes import Axes, SubplotBase

from ..insight.metrics import kolmogorov_smirnov_distance, earth_movers_distance


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


def axes_grid(ax: Union[Axes, SubplotBase], rows: int, cols: int,
              col_titles: List[str] = None, row_titles: List[str] = None, **kwargs):
    """

    Args:
        ax: The axes to subdivide.
        rows: number of rows.
        cols: Number of columns.
        col_titles: Title for each column.
        row_titles: Title for each row.
        **kwargs: wspace, hspace, height_ratios, width_ratios.

    """
    col_titles = col_titles or ['' for _ in range(cols)]
    row_titles = row_titles or ['' for _ in range(rows)]
    ax.set_axis_off()
    sp_spec = ax.get_subplotspec()
    sgs = sp_spec.subgridspec(rows, cols, **kwargs)
    fig = ax.figure
    col_axes: List[mpl.axes.Axes] = list()
    for c in range(cols):
        sharey = col_axes[0] if c > 0 else None
        ax = fig.add_subplot(sgs[:, c], sharey=sharey)
        ax.set_title(col_titles[c])
        col_axes.append(ax)

    if rows == 1:
        col_axes[0].set_ylabel(row_titles[0])
        return col_axes
    else:
        for col_ax in col_axes:
            col_ax.set_axis_off()

        axes = []
        for r in range(rows):
            if cols == 1:
                if r == 0:
                    axes.append(fig.add_subplot(sgs[r, 0]))
                else:
                    axes.append(fig.add_subplot(sgs[r, 0], sharex=axes[0]))
                axes[r].set_ylabel(row_titles[r])
            else:
                row_axes = list()
                if r == 0:
                    row_axes.append(fig.add_subplot(sgs[r, 0]))
                else:
                    row_axes.append(fig.add_subplot(sgs[r, 0], sharex=axes[0][0]))
                row_axes[0].set_ylabel(row_titles[r])
                for c in range(1, cols):
                    if r == 0:
                        row_axes.append(fig.add_subplot(sgs[r, c], sharey=row_axes[0]))
                    else:
                        row_axes.append(fig.add_subplot(sgs[r, c], sharex=axes[0][c], sharey=row_axes[0]))
                axes.append(row_axes)

    return axes


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
    plt.show()


def plot_first_order_metric_distances(result: pd.Series, metric_name: str):
    if len(result) == 0:
        return

    df = pd.DataFrame(result.dropna()).reset_index()

    plt.figure(figsize=(8, int(len(df) / 2) + 2))
    g = sns.barplot(y='index', x=metric_name, data=df)
    g.set_xlim(0.0, 1.0)
    plt.title(f'{metric_name}s')
    plt.show()


def plot_second_order_metric_matrix(matrix: pd.DataFrame, title: str = None, ax=None, symmetric=True):
    # Generate a mask for the upper triangle
    if symmetric:
        mask = np.zeros_like(matrix, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
    else:
        mask = None

    sns.set(style='white')
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    hm = sns.heatmap(matrix, mask=mask, cmap=cmap, vmin=-1.0, vmax=1.0, center=0,
                     square=True, linewidths=.5, cbar=False, ax=ax, annot=True, fmt='.2f')

    if ax:
        ax.set_ylim(ax.get_ylim()[0] + .5, ax.get_ylim()[1] - .5)
        ax.label_outer()
    if title:
        hm.set_title(title)


def plot_second_order_metric_matrices(
        matrix_test: pd.DataFrame, matrix_synth: pd.DataFrame,
        metric_name: str, symmetric=True
):
    if len(matrix_test.columns) == 0:
        return

    # Set up the matplotlib figure
    scale = 1.0
    label_length = len(max(matrix_test.columns, key=len))*0.08
    width, height = (2*scale*len(matrix_test.columns)+label_length, scale*len(matrix_test) + label_length)
    figsize = (width, height)

    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    fig.suptitle(f"{metric_name} Matrices", fontweight='bold', fontsize=14)
    ax1, ax2 = axes_grid(ax, rows=1, cols=2, col_titles=['Original', 'Synthetic'], wspace=1/len(matrix_test.columns))

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
        logger.error('Column {} cant be shown :: {}'.format(col_test.name, e))

    ax.set_title(title)
    plt.legend()


def plot_time_series(x, t, ax):
    kind = x.dtype.kind
    if kind in {"i", "f"}:
        sequence_line_plot(x=x, t=t, ax=ax)
    else:
        sequence_index_plot(x=x, t=t, ax=ax)


def plot_continuous_time_series(df_orig, df_synth, df_test, col, identifiers=None, ax=None):
    ax = ax or plt.gca()
    if identifiers is not None:
        axes = axes_grid(
            ax, len(identifiers), 1, col_titles=['', ''], row_titplot_second_order_metric_matricesles=identifiers,
            wspace=0, hspace=0
        )

        for j, idf in enumerate(identifiers[:5]):
            x = pd.to_numeric(df_orig.loc[:, (idf, col)], errors='coerce').dropna()
            if len(x) > 1:
                axes[j].plot(x.index, x.values, color=COLOR_ORIG, label='orig')

            x = pd.to_numeric(df_synth.loc[:, (idf, col)], errors='coerce').dropna()
            if len(x) > 1:
                axes[j].plot(x.index, x.values, color=COLOR_SYNTH, label='synth', linewidth=1)

            x = pd.to_numeric(df_test.loc[:, (idf, col)], errors='coerce').dropna()
            if len(x) > 1:
                axes[j].axvspan(x.index[0], x.index[-1], facecolor='0.1', alpha=0.02)
                axes[j].plot(x.index, x.values, color=COLOR_ORIG, linestyle='dashed', linewidth=1, label='test')

            axes[j].label_outer()
        axes[0].legend()
    else:
        orig_ax, synth_ax = axes_grid(
            ax, 1, 2, col_titles=['Original', 'Synthetic'], row_titles=[''], wspace=0, hspace=0
        )
        x = pd.to_numeric(df_orig[col], errors='coerce').dropna()
        if len(x) > 1:
            orig_ax.plot(x.index, x.values, color=COLOR_ORIG)

        x = pd.to_numeric(df_synth[col], errors='coerce').dropna()
        if len(x) > 1:
            synth_ax.plot(x.index, x.values, color=COLOR_SYNTH)


def plot_categorical_time_series(df_orig, df_synth, col, identifiers=None, ax=None):
    if identifiers is not None:
        ax.set_axis_off()
        fig = ax.figure
        sp_spec = ax.get_subplotspec()
        gsgs = sp_spec.subgridspec(len(identifiers), 2, hspace=0, wspace=0)

        orig_ax = fig.add_subplot(gsgs[:, 0])
        orig_ax.set_title('Original')
        orig_ax.set_axis_off()
        synth_ax = fig.add_subplot(gsgs[:, 1])
        synth_ax.set_title('Synthetic')
        synth_ax.set_axis_off()

        for j, idf in enumerate(identifiers):
            ax = fig.add_subplot(gsgs[j, 0])
            x = df_orig.loc[df_orig[identifiers.name] == idf, col].dropna().values
            oh = OneHotEncoder(dtype=np.float32, sparse=False)
            data = oh.fit_transform(x.reshape(-1, 1))
            cmap = cm.colors.LinearSegmentedColormap.from_list('orig', colors=['#FFFFFF', COLOR_ORIG])
            ax.imshow(data.T, cmap=cmap, aspect='auto')
            ax.set_ylabel(idf)
            ax.get_yaxis().set_ticks(np.arange(0, len(oh.categories_[0]), 1))
            ax.set_yticklabels(oh.categories_[0])
            ax.label_outer()
            ax.autoscale_view()

            ax2 = fig.add_subplot(gsgs[j, 1])
            x = df_synth.loc[df_synth[identifiers.name] == idf, col].dropna().values
            oh = OneHotEncoder(dtype=np.float32, sparse=False)
            data = oh.fit_transform(x.reshape(-1, 1))
            cmap = cm.colors.LinearSegmentedColormap.from_list('orig', colors=['#FFFFFF', COLOR_SYNTH])
            ax2.imshow(data.T, cmap=cmap, aspect='auto')
            ax2.set_ylabel(idf)
            ax2.get_yaxis().set_ticks(np.arange(0, len(oh.categories_[0]), 1))
            ax2.set_yticklabels(oh.categories_[0])
            ax2.label_outer()
            ax2.autoscale_view()


def plot_cross_correlations(df_orig, df_synth, col, identifiers=None, max_order=100, ax=None):
    ax = ax or plt.gca()
    if identifiers is not None:
        axes = axes_grid(
            ax, len(identifiers), len(identifiers), identifiers, identifiers, wspace=0.0, hspace=0.0
        )
        for i, idx in enumerate(identifiers):
            for j, idy in enumerate(identifiers):
                ax = axes[i][j]
                auto_corr_orig = scaled_correlation(
                    df_orig.loc[:, (idx, col)].values, df_orig.loc[:, (idy, col)].values, window=20, lags=100)
                auto_corr_synth = scaled_correlation(
                    df_synth.loc[:, (idx, col)].values, df_synth.loc[:, (idy, col)].values, window=20, lags=100)
                lags = np.array(range(100))

                ax.bar(lags, auto_corr_orig, width=1.0, color=COLOR_ORIG, alpha=0.5)
                ax.bar(lags, auto_corr_synth, width=1.0, color=COLOR_SYNTH, alpha=0.5)
                ax.label_outer()
    else:
        auto_corr_orig = scaled_correlation(
            df_orig.loc[:, col].values, df_orig.loc[:, col].values, window=20, lags=max_order)
        auto_corr_synth = scaled_correlation(
            df_synth.loc[:, col].values, df_synth.loc[:, col].values, window=20, lags=max_order)
        lags = np.array(range(100))

        ax.bar(lags, auto_corr_orig, width=1.0, color=COLOR_ORIG, alpha=0.7)
        ax.bar(lags, auto_corr_orig, width=1.0, color=COLOR_SYNTH, alpha=0.7)
        ax.label_outer()


def scaled_correlation(x, y, window=20, lags=100):
    W = window
    L = lags

    not_nan = np.logical_not(np.logical_or(np.isnan(x), np.isnan(y)))
    x = x[not_nan]
    y = y[not_nan]

    num = math.floor((len(x) - L) / W)

    auto_corr = np.mean(np.array([
        [np.corrcoef(
            np.stack((x[n * W + lag:(n + 1) * W + lag], y[n * W:(n + 1) * W]), axis=0)
        )[0][1] for lag in range(L)]
        for n in range(num)
    ]), axis=0)

    return auto_corr


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


def plt_dist_orig_snyh(df_orig: pd.DataFrame, df_synth: pd.DataFrame, key: str, unique_threshold: int = 30,
                       ax: plt.Axes = None) -> float:

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

        orig_vc = orig_vc[key].values[(start <= orig_vc) & (orig_vc <= end)]
        synth_vc = synth_vc[key].values[(start <= synth_vc) & (synth_vc <= end)]

        sns.distplot(orig_vc[key], color=COLOR_ORIG, label='orig', ax=ax)
        sns.distplot(synth_vc[key], color=COLOR_SYNTH, label='synth', ax=ax)
        dist = kolmogorov_smirnov_distance(orig_vc, synth_vc, key)

    else:
        # We sample orig and synth them so that they have the same size to make the plots more comprehensive
        sample_size = min(len(orig_vc), len(synth_vc))
        concatenated = pd.concat([orig_vc.assign(dataset='orig').sample(sample_size),
                                  synth_vc.assign(dataset='synth').sample(sample_size)])

        ax = sns.countplot(x=key, hue='dataset', data=concatenated,
                           palette={'orig': COLOR_ORIG, 'synth': COLOR_SYNTH}, ax=ax)
        dist = earth_movers_distance(orig_vc, synth_vc, key)

    ax.legend()
    return float(dist or 0.0)
