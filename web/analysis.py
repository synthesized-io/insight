from abc import ABC
from io import StringIO
from typing import Iterable

import numpy as np
import pandas as pd
from numpy.lib import histograms
from sklearn.neighbors import KernelDensity

from synthesized.core import BasicSynthesizer
from synthesized.core.values import ContinuousValue

REMOVE_OUTLIERS = 0.01
DEFAULT_KERNEL = 'epanechnikov'
DEFAULT_MAX_BINS = 50


class ColumnMeta(ABC):
    def __init__(self, name: str, type: str, plot_type: str):
        self.name = name
        self.type = type
        self.plot_type = plot_type


class ContinuousPlotData:
    def __init__(self, edges: Iterable[float], hist: Iterable[int], density: Iterable[float]):
        self.edges = edges
        self.hist = hist
        self.density = density


class CategoricalPlotData:
    def __init__(self, bins: Iterable[str], hist: Iterable[int]):
        self.bins = bins
        self.hist = hist


class ContinuousMeta(ColumnMeta):
    def __init__(self, name: str, type: str, mean: float, std: float, median: float, min: float, max: float,
                 n_nulls: int, plot_data: ContinuousPlotData) -> None:
        super().__init__(name, type, "density")
        self.mean = mean
        self.std = std
        self.median = median
        self.min = min
        self.max = max
        self.n_nulls = n_nulls
        self.plot_data = plot_data


class CategoricalMeta(ColumnMeta):
    def __init__(self, name: str, type: str, n_unique: int, most_frequent: str, most_occurrences: int,
                 plot_data: CategoricalPlotData) -> None:
        super().__init__(name, type, "histogram")
        self.n_unique = n_unique
        self.most_frequent = most_frequent
        self.most_occurrences = most_occurrences
        self.plot_data = plot_data


class DatasetMeta:
    def __init__(self, n_rows: int, n_columns: int, n_types: int, columns: Iterable[ColumnMeta]) -> None:
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.n_types = n_types
        self.columns = columns


# Compute dataset's meta from scratch
def extract_dataset_meta(data: pd.DataFrame, remove_outliers: int=REMOVE_OUTLIERS) -> DatasetMeta:
    raw_data = StringIO()
    data.to_csv(raw_data, index=False)
    data_wo_nans = data.dropna()
    synthesizer = BasicSynthesizer(data=data_wo_nans.sample(1000))
    value_types = set()
    columns_meta = []
    for value in synthesizer.values:
        value_types.add(str(value))
        if isinstance(value, ContinuousValue):
            q = [remove_outliers / 2., 1 - remove_outliers / 2.]
            start, end = np.quantile(data_wo_nans[value.name], q)
            column_cleaned = data_wo_nans[(data_wo_nans[value.name] > start) & (data_wo_nans[value.name] < end)][value.name]
            bins = _bounded_bin_selector(column_cleaned, max_bins=DEFAULT_MAX_BINS)
            hist, edges = np.histogram(column_cleaned, bins=bins)
            kde = KernelDensity(kernel=DEFAULT_KERNEL)
            log_density = kde.fit(column_cleaned.values.reshape(-1, 1)).score_samples(edges.reshape(-1, 1))
            density = np.exp(log_density)

            hist = list(map(int, hist))
            edges = list(map(float, edges))
            density = list(map(float, density))

            columns_meta.append(ContinuousMeta(
                name=value.name,
                type=str(value),
                mean=float(data[value.name].mean()),
                std=float(data[value.name].std()),
                median=float(data[value.name].median()),
                min=float(data[value.name].min()),
                max=float(data[value.name].max()),
                n_nulls=int(data[value.name].isnull().sum()),
                plot_data=ContinuousPlotData(
                    hist=hist,
                    edges=edges,
                    density=density
                )
            ))
        else:
            most_frequent = data[value.name].value_counts().idxmax()
            bins = sorted(data[value.name].dropna().unique())
            counts = data[value.name].value_counts().to_dict()
            hist = [counts[x] for x in bins]

            hist = list(map(int, hist))
            bins = list(map(str, bins))

            columns_meta.append(CategoricalMeta(
                name=value.name,
                type=str(value),
                n_unique=int(data[value.name].nunique()),
                most_frequent=str(most_frequent),
                most_occurrences=int(len(data[data[value.name] == most_frequent])),
                plot_data=CategoricalPlotData(
                    hist=hist,
                    bins=bins
                )
            ))
    return DatasetMeta(
        n_rows=len(data),
        n_columns=len(data.columns),
        n_types=len(value_types),
        columns=columns_meta,
    )


# Use existing dataset meta to create a new one
def recompute_dataset_meta(data: pd.DataFrame, meta: DatasetMeta) -> DatasetMeta:
    data_wo_nans = data.dropna()
    raw_data = StringIO()
    data.to_csv(raw_data, index=False)
    columns_meta = []
    for column_meta in meta.columns:
        if column_meta.plot_type == 'density':  # we want duck typing here
            column_meta: ContinuousMeta = column_meta
            column_cleaned = data_wo_nans[column_meta.name]
            hist, edges = np.histogram(column_cleaned, bins=column_meta.plot_data.edges)
            kde = KernelDensity(kernel=DEFAULT_KERNEL)
            log_density = kde.fit(column_cleaned.values.reshape(-1, 1)).score_samples(edges.reshape(-1, 1))
            density = np.exp(log_density)

            hist = list(map(int, hist))
            edges = list(map(float, edges))
            density = list(map(float, density))

            columns_meta.append(ContinuousMeta(
                name=column_meta.name,
                type=column_meta.type,
                mean=float(data[column_meta.name].mean()),
                std=float(data[column_meta.name].std()),
                median=float(data[column_meta.name].median()),
                min=float(data[column_meta.name].min()),
                max=float(data[column_meta.name].max()),
                n_nulls=int(data[column_meta.name].isnull().sum()),
                plot_data=ContinuousPlotData(
                    hist=hist,
                    edges=edges,
                    density=density
                )
            ))
        elif column_meta.plot_type == 'histogram':  # we want duck typing here
            column_meta: CategoricalMeta = column_meta
            most_frequent = data[column_meta.name].value_counts().idxmax()
            bins = column_meta.plot_data.bins
            counts = data[column_meta.name].value_counts().to_dict()
            counts = {str(k): v for k, v in counts.items()}  # bins in original meta are always strings
            hist = [counts.get(x, 0) for x in bins]
            hist = list(map(int, hist))
            columns_meta.append(CategoricalMeta(
                name=column_meta.name,
                type=column_meta.type,
                n_unique=int(data[column_meta.name].nunique()),
                most_frequent=str(most_frequent),
                most_occurrences=int(len(data[data[column_meta.name] == most_frequent])),
                plot_data=CategoricalPlotData(
                    hist=hist,
                    bins=bins
                )
            ))
    return DatasetMeta(
        n_rows=len(data),
        n_columns=len(data.columns),
        n_types=meta.n_types,
        columns=columns_meta,
    )


def _bounded_bin_selector(a, max_bins):
    first_edge, last_edge = histograms._get_outer_edges(a, range=None)
    width = histograms._hist_bin_auto(a, (first_edge, last_edge))
    if width:
        bins_auto = int(np.ceil(histograms._unsigned_subtract(last_edge, first_edge) / width))
    else:
        bins_auto = 1
    return min(bins_auto, max_bins)
