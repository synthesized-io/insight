from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..check import Check, ColumnCheck


def zipped_hist(
    data: Tuple[pd.Series, ...],
    check: Check = ColumnCheck(),
    bin_edges: Optional[np.ndarray] = None,
    normalize: bool = True,
    ret_bins: bool = False,
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
    is_continuous = check.continuous(pd.Series(joint))

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


def bootstrap_statistic(data: Union[Tuple[pd.Series], Tuple[pd.Series, pd.Series]],
                        statistic: Union[Callable[[pd.Series, pd.Series], float], Callable[[pd.Series], float]],
                        n_samples: int = 1000, sample_size=None) -> np.ndarray:
    """
    Compute the samples of a statistic estimate using the bootstrap method.

    Args:
        data: Data on which to compute the statistic.
        statistic: Function that computes the statistic.
        n_samples: Optional; Number of bootstrap samples to perform.

    Returns:
        The bootstrap samples.
    """
    if sample_size is None:
        sample_size = max((len(x) for x in data))

    def get_sample_idx(x):
        return np.random.randint(0, len(x), min(len(x), sample_size))

    statistic_samples = np.empty(n_samples)
    for i in range(n_samples):
        sample_idxs = [get_sample_idx(x) for x in data]
        statistic_samples[i] = statistic(*[x[idx] for x, idx in zip(data, sample_idxs)])
    return statistic_samples


def bootstrap_binned_statistic(data: Tuple[pd.Series, pd.Series], statistic: Callable[[pd.Series, pd.Series], float],
                               n_samples: int = 1000) -> np.ndarray:
    """
    Compute the samples of a binned statistic estimate using the bootstrap method.

    Args:
        data: Data for which to compute the statistic.
        statistic: Function that computes the statistic.
        n_samples: Optional; Number of bootstrap samples to perform.

    Returns:
        The bootstrap samples.
    """

    statistic_samples = np.empty(n_samples)

    with np.errstate(divide='ignore', invalid='ignore'):
        p_x = np.nan_to_num(data[0] / data[0].sum())
        p_y = np.nan_to_num(data[1] / data[1].sum())

    n_x = data[0].sum()
    n_y = data[1].sum()

    x_samples = np.random.multinomial(n_x, p_x, size=n_samples)
    y_samples = np.random.multinomial(n_y, p_y, size=n_samples)

    for i in range(n_samples):
        statistic_samples[i] = statistic(x_samples[i], y_samples[i])

    return statistic_samples
