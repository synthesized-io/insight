from typing import List, Union

import matplotlib as mpl
from matplotlib.axes import Axes, SubplotBase


def axes_grid(
        ax: Union[Axes, SubplotBase], rows: int, cols: int, col_titles: List[str] = None,
        row_titles: List[str] = None, sharey=True, **kwargs
) -> Union[List[Union[Axes, SubplotBase]], List[List[Union[Axes, SubplotBase]]]]:
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
        sy = None
        if sharey and c > 0:
            sy = col_axes[0]

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
    for c in range(1, cols):
        row_axes.append(fig.add_subplot(sgs[0, c], sharey=row_axes[0]))
    axes.append(row_axes)

    for r in range(1, rows):
        row_axes = list()
        row_axes.append(fig.add_subplot(sgs[r, 0], sharex=axes[0][0]))
        row_axes[0].set_ylabel(row_titles[r])
        for c in range(1, cols):
            row_axes.append(
                fig.add_subplot(sgs[r, c], sharex=axes[0][c], sharey=row_axes[0])
            )
        axes.append(row_axes)

    return axes
