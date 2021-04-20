from typing import Any, Dict, Generic, Optional, Sequence, TypeVar, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .kde import KernelDensityEstimate
from ..base import DiscreteModel
from ..exceptions import ModelNotFittedError
from ...metadata import Affine, AType, ExtractionError, Nominal, NType, SType, ValueMeta

HistogramType = TypeVar('HistogramType', bound='Histogram')


class Histogram(DiscreteModel[Nominal[NType, ValueMeta], NType], Generic[NType]):
    """A Histogram used to model a discrete variable

    Attributes:
        name (str): The name of the data column that this histogram models.
        categories (MutableSequence[NType]): A list of the categories/bins (left edges if bin_width is not None).
        probabilities (Dict[NType, float]): A mapping of each of the categories to a probability.

    """

    def __init__(
            self, meta: Nominal[NType, ValueMeta], probabilities: Optional[Dict[NType, float]] = None,
    ):
        super().__init__(meta=meta)  # type: ignore
        self._probabilities: Optional[Dict[NType, float]] = probabilities

        if probabilities is not None:
            self._fitted = True

    @property
    def probabilities(self) -> Dict[NType, float]:
        if self._probabilities is None:
            raise ModelNotFittedError
        return self._probabilities

    @probabilities.setter
    def probabilities(self, probs: Dict[NType, float]) -> None:
        self._probabilities = probs

    @property
    def dtype(self) -> str:
        if self.bin_width is not None:
            return f"interval[{self._meta.dtype}]"
        else:
            return self._meta.dtype

    @property
    def bin_width(self) -> Union[None, SType]:
        if len(self._meta.categories) < 2:
            return None

        if not isinstance(self._meta, Affine):
            return None
        mn, mx = self._meta.min, self._meta.max

        if mn is None or mx is None:
            return None

        bin_width: SType = self._meta.unit_meta.precision
        smallest_diff: SType = np.diff(self._meta.categories).min()

        if bin_width > smallest_diff and bin_width.astype(np.float64) > 0:
            return bin_width

        return None

    @property
    def categories(self) -> Union[Sequence[NType], Sequence[pd.IntervalDtype]]:
        bin_width = self.bin_width

        if isinstance(self._meta, Affine) and bin_width is not None:
            assert self._meta.min is not None and self._meta.max is not None
            rng = self._meta.max - self._meta.min
            # makes sure we include the max
            rng_max = self._meta.max + bin_width if (rng % bin_width).item() != 0 else self._meta.max
            categories: Union[Sequence[NType], Sequence[pd.IntervalDtype[NType]]] = pd.interval_range(
                self._meta.min, rng_max, freq=bin_width.item(), closed='left'  # type: ignore
            )
        else:
            categories = self._meta.categories

        return categories

    def fit(self: HistogramType, df: pd.DataFrame) -> HistogramType:
        super().fit(df=df)
        if len(self.categories) == 0:
            self.probabilities = {}

        if isinstance(self.categories, pd.IntervalIndex):
            cut = pd.cut(df[self.name], bins=self.categories)
            value_counts = pd.value_counts(cut, normalize=True, dropna=True, sort=False)
        else:
            value_counts = pd.value_counts(df[self.name], normalize=True, dropna=True, sort=False)

        self.probabilities = {cat: value_counts.get(cat, 0.0) for cat in self.categories}

        return self

    def sample(self, n: int, produce_nans: bool = False, conditions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        df = pd.DataFrame(
            {self.name: np.random.choice([*self.probabilities.keys()], size=n, p=[*self.probabilities.values()])})

        if produce_nans:
            df[self.name] = self.add_nans(df[self.name], self.nan_freq)

        return df

    def probability(self, x: Any) -> float:
        if x is None and self.nan_freq is not None:
            return self.nan_freq

        if x in self.probabilities:
            prob: float = self.probabilities[x]
        else:
            prob = 0.0

        return prob

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "probabilities": self._probabilities
        })

        return d

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> 'Histogram':
        meta_dict = cast(Dict[str, object], d["meta"])
        meta = Nominal[NType, ValueMeta].from_dict(meta_dict)
        model = cls(meta=meta, probabilities=cast(Optional[Dict[NType, float]], d["probabilities"]))
        model._fitted = cast(bool, d["fitted"])

        return model

    @classmethod
    def bin_affine_meta(cls, meta: Affine[AType], max_bins: int = 20) -> 'Histogram[pd.IntervalDtype[AType]]':

        if meta.max is not None and meta.min is not None:
            rng = meta.max - meta.min

            if meta.dtype == 'M8[ns]':
                dtype = 'interval[M8[ns]]'  # TODO: find away to handle 'interval[M8[D]]' instead of 'interval[M8[ns]]'
            else:
                dtype = f'interval[{meta.dtype}]'

            if (rng / max_bins) > meta.unit_meta.precision:
                categories = pd.interval_range(meta.min, meta.max, periods=max_bins, closed='left').astype(dtype)
            else:
                bin_width = meta.unit_meta.precision
                categories = pd.interval_range(meta.min, meta.max, freq=bin_width.item(), closed='left')

        else:
            raise ExtractionError("Meta doesn't have both max and min defined.")

        if isinstance(meta, KernelDensityEstimate) and meta._fitted:
            norm = meta.integrate(meta.min, meta.max)
            probabilities: Optional[Dict[AType, float]] = {
                c: meta.integrate(np.array(c.left, dtype=meta.dtype), np.array(c.right, dtype=meta.dtype)) / norm

                for c in categories
            }
        else:
            probabilities = None

        binned_meta = Nominal[pd.IntervalDtype, ValueMeta](
            meta.name, categories=categories, nan_freq=meta.nan_freq, num_rows=meta.num_rows)
        binned_meta.dtype = dtype
        hist = Histogram(meta=binned_meta, probabilities=probabilities)

        return hist

    def plot(self, **kwargs) -> plt.Figure:

        fig = plt.Figure(figsize=(7, 4))
        sns.set_theme(**kwargs)

        plot_data = pd.DataFrame({
            self.name: self.categories,
            'probability': [self.probability(c) for c in self.categories]
        })
        sns.barplot(data=plot_data, x=self.name, y='probability', ax=fig.gca())

        for tick in fig.gca().get_xticklabels():
            tick.set_rotation(90)

        np.datetime64()

        return fig
