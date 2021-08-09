from typing import Tuple, Optional, Union
from enum import Enum

import numpy as np
import pandas as pd

from src.synthesized_insight.check import ColumnCheck


class DistrType(Enum):
    """Indicates the type distribution of data in a series."""

    Continuous = "continuous"
    Binary = "binary"
    Categorical = "categorical"
    Datetime = "datetime"

    def is_continuous(self):
        return self == DistrType.Continuous

    def is_binary(self):
        return self == DistrType.Binary

    def is_categorical(self):
        return self == DistrType.Categorical

    def is_datetime(self):
        return self == DistrType.Datetime


def infer_distr_type(column: pd.Series,
                     ctl_mult: float = 2.5,
                     min_num_unique: int = 10) -> DistrType:
    """Infers whether the data in a column or series is datetime, continuous, categorical or binary.
    Args:
        column (pd.Series):
            The column from the data or data series to consider.
        ctl_mult (float, optional):
            Categorical threshold log multiplier. Defaults to 2.5.
        min_num_unique (int, optional):
            Minimum number of unique values for the data to be continuous. Defaults to 10.
    Returns:
        DistrType:
            The output is an enum representing the type of distribution.
    Examples:
        >>> col_type = infer_distr_type(range(1000))
        >>> col_type.is_continuous()
        True
        >>> col_type.is_binary()
        False
    """

    check = ColumnCheck()
    col = check.infer_dtype(column)

    unique = col.unique()
    n_unique = len(unique)
    n_rows = len(col)
    dtype = col.dtype

    if n_unique == 2:
        return DistrType.Binary

    elif dtype == "float64":
        return DistrType.Continuous

    elif dtype == "datetime64[ns]":
        return DistrType.Datetime

    elif n_unique > max(min_num_unique, ctl_mult * np.log(n_rows)) and dtype in ["float64", "int64"]:
        return DistrType.Continuous

    else:
        return DistrType.Categorical


def zipped_hist(
    data: Tuple[pd.Series, ...],
    bin_edges: Optional[np.ndarray] = None,
    normalize: bool = True,
    ret_bins: bool = False,
    distr_type: Optional[str] = None,
) -> Union[Tuple[pd.Series, ...], Tuple[Tuple[pd.Series, ...], Optional[np.ndarray]]]:
    """Bins a tuple of series' and returns the aligned histograms.
    Args:
        data (Tuple[pd.Series, ...]):
            A tuple consisting of the series' to be binned. All series' must have the same dtype.
        bin_edges (Optional[np.ndarray], optional):
            Bin edges to bin continuous data by. Defaults to None.
        normalize (bool, optional):
            Normalize the histograms, turning them into pdfs. Defaults to True.
        ret_bins (bool, optional):
            Returns the bin edges used in the histogram. Defaults to False.
        distr_type (Optional[str]):
            The type of distribution of the target attribute. Can be "categorical" or "continuous".
            If None the type of distribution is inferred based on the data in the column.
            Defaults to None.
    Returns:
        Union[Tuple[np.ndarray, ...], Tuple[Tuple[np.ndarray, ...], Optional[np.ndarray]]]:
            A tuple of np.ndarrays consisting of each histogram for the input data.
            Additionally returns bins if ret_bins is True.
    """

    joint = pd.concat(data)
    is_continuous = distr_type == "continuous" if distr_type is not None else infer_distr_type(joint).is_continuous()

    # Compute histograms of the data, bin if continuous
    if is_continuous:
        # Compute shared bin_edges if not given, and use np.histogram to form histograms
        if bin_edges is None:
            bin_edges = np.histogram_bin_edges(joint, bins="auto")

        hists = [np.histogram(series, bins=bin_edges)[0] for series in data]

        if normalize:
            with np.errstate(divide="ignore", invalid="ignore"):
                hists = [np.nan_to_num(hist / hist.sum()) for hist in hists]

    else:
        # For categorical data, form histogram using value counts and align
        space = joint.unique()

        dicts = [sr.value_counts(normalize=normalize) for sr in data]
        hists = [np.array([d.get(val, 0) for val in space]) for d in dicts]

    ps = [pd.Series(hist) for hist in hists]

    if ret_bins:
        return tuple(ps), bin_edges

    return tuple(ps)
