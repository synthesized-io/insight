import logging
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.axes import Axes, SubplotBase

from .style import COLOR_ORIG, COLOR_SYNTH
from ...util import axes_grid

logger = logging.getLogger(__name__)


def plot_time_series(x, t, ax):
    kind = x.dtype.kind
    if kind in {"i", "f"}:
        sequence_line_plot(x=x, t=t, ax=ax)
    else:
        sequence_index_plot(x=x, t=t, ax=ax)


def plot_series(sr: pd.Series, ax: Union[Axes, SubplotBase] = None, **kwargs):
    ax = ax or plt.gca()
    x = pd.to_numeric(sr, errors='coerce').dropna()
    if len(x) > 1:
        ax.plot(x.index, x.values, **kwargs)


def plot_continuous_time_series(df_orig: pd.DataFrame, df_synth: pd.DataFrame, col: str, forecast_from=None,
                                identifiers=None, ax: Union[Axes, SubplotBase] = None):
    ax = ax or plt.gca()
    if identifiers is not None:
        axes: List[Union[Axes, SubplotBase]] = axes_grid(
            ax, len(identifiers), 1, col_titles=['', ''], row_titles=identifiers, wspace=0, hspace=0
        )

        for j, idf in enumerate(identifiers):
            plot_series(sr=df_orig.xs(idf).loc[:forecast_from, col], ax=axes[j], color=COLOR_ORIG, label='orig')
            plot_series(sr=df_synth.xs(idf)[col], ax=axes[j], color=COLOR_SYNTH, label='synth')

            if forecast_from is not None:
                sr = df_orig.xs(idf).loc[forecast_from:, col]
                plot_series(sr=sr, ax=axes[j], color=COLOR_ORIG, linestyle='dashed', linewidth=1, label='test')
                axes[j].axvspan(sr.index[0], sr.index[-1], facecolor='0.1', alpha=0.02)

            axes[j].label_outer()
        axes[0].legend()
    else:
        orig_ax, synth_ax = axes_grid(
            ax, 1, 2, col_titles=['Original', 'Synthetic'], row_titles=[''], wspace=0, hspace=0
        )
        assert isinstance(orig_ax, Axes)
        assert isinstance(synth_ax, Axes)

        x = pd.to_numeric(df_orig[col], errors='coerce').dropna()
        if len(x) > 1:
            orig_ax.plot(x.index, x.values, color=COLOR_ORIG)

        x = pd.to_numeric(df_synth[col], errors='coerce').dropna()
        if len(x) > 1:
            synth_ax.plot(x.index, x.values, color=COLOR_SYNTH)


def plot_categorical_time_series(df_orig, df_synth, col, identifiers=None, ax: Union[Axes, SubplotBase] = None):
    raise NotImplementedError


def sequence_index_plot(x, t, ax: Axes, cmap_name: str = "YlGn"):
    values = np.unique(x)
    val2idx = {val: i for i, val in enumerate(values)}
    cmap = cm.get_cmap(cmap_name)
    colors = [cmap(j / values.shape[0]) for j in range(values.shape[0])]

    for i, val in enumerate(x):
        ax.fill_between((i, i + 1), 2, facecolor=colors[val2idx[val]])
    ax.get_yaxis().set_visible(False)


def sequence_line_plot(x, t, ax):
    sns.lineplot(x=t, y=x, ax=ax)
