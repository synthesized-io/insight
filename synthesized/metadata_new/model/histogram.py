from typing import Generic, Optional, Dict, Any, cast, Sequence, overload, Union, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .kde import KernelDensityEstimate
from ..base import DiscreteModel
from ..base.value_meta import NType, Nominal, Affine, AType
from ..exceptions import MetaNotExtractedError, ModelNotFittedError, ExtractionError


class Histogram(DiscreteModel[NType], Generic[NType]):
    """A Histogram used to model a discrete variable

    Attributes:
        name (str): The name of the data column that this histogram models.
        categories (MutableSequence[NType]): A list of the categories/bins (left edges if bin_width is not None).
        probabilities (Dict[NType, float]): A mapping of each of the categories to a probability.

    """

    def __init__(
            self, name: str, categories: Optional[Sequence[NType]] = None, nan_freq: Optional[float] = None,
            probabilities: Optional[Dict[NType, float]] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)  # type: ignore
        if self.categories is not None:
            self._extracted = True
        self.probabilities: Optional[Dict[NType, float]] = probabilities
        if self.probabilities is not None:
            self._fitted = True

    def fit(self: 'Histogram[NType]', df: pd.DataFrame) -> 'Histogram[NType]':
        super().fit(df=df)
        assert self.categories is not None
        if isinstance(self.categories, pd.IntervalIndex):
            cut = pd.cut(df[self.name], bins=self.categories)
            value_counts = pd.value_counts(cut, normalize=True, dropna=False, sort=False)
        else:
            value_counts = pd.value_counts(df[self.name], normalize=True, dropna=False, sort=False)

        self.probabilities = {cat: value_counts.get(cat, 0.0) for cat in self.categories}
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if not self._fitted:
            raise ModelNotFittedError

        assert self.probabilities is not None
        samples = np.random.choice([*self.probabilities.keys()], size=n, p=[*self.probabilities.values()])
        return pd.DataFrame({self.name: samples})

    def probability(self, x: Any) -> float:
        if not self._extracted:
            raise MetaNotExtractedError
        if not self._fitted:
            raise ModelNotFittedError

        self.probabilities = cast(Dict[NType, float], self.probabilities)

        if x is None:
            return cast(float, self.nan_freq)

        x = cast(NType, x)

        if x in self.probabilities:
            prob: float = self.probabilities[x]
        else:
            prob = 0.0

        return prob

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "probabilities": self.probabilities
        })
        return d

    @classmethod
    def bin_affine_meta(cls, meta: Affine[AType], max_bins: int = 20) -> 'Histogram[pd.IntervalDtype[AType]]':

        if meta.max is not None and meta.min is not None:
            rng = meta.max - meta.min

            if meta.dtype == 'M8[D]':
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
            norm = meta.integrated_probability(meta.min, meta.max)
            probabilities: Optional[Dict[AType, float]] = {
                c: meta.integrated_probability(c.left.to_numpy(), c.right.to_numpy())/norm
                for c in categories
            }
        else:
            probabilities = None

        hist = Histogram(name=meta.name, categories=categories, nan_freq=meta.nan_freq, probabilities=probabilities)
        hist.dtype = dtype

        return hist

    @classmethod
    @overload
    def from_meta(cls: Type['Histogram'], meta: Affine[AType]) -> 'Union[Histogram[AType], Histogram[pd.IntervalDtype[AType]]]':
        ...

    @classmethod
    @overload
    def from_meta(cls: Type['Histogram'], meta: Nominal[NType]) -> 'Histogram[NType]':
        ...

    @classmethod
    def from_meta(cls: Type['Histogram'], meta: Nominal[NType]) -> 'Union[Histogram[NType], Histogram[pd.IntervalDtype[AType]]]':

        dtype = meta.dtype
        probabilities = None

        if isinstance(meta, Affine) and meta.max is not None and meta.min is not None and meta.max != meta.min:
            rng = meta.max - meta.min
            bin_width = meta.unit_meta.precision
            smallest_diff = np.diff(meta.categories).min()

            if bin_width > smallest_diff:
                try:
                    rng / bin_width
                    categories = pd.interval_range(meta.min, meta.max, freq=bin_width.item(), closed='left')
                    dtype = str(categories.dtype)  # TODO: find way for 'interval[M8[D]]' instead of 'interval[M8[ns]]'
                    if isinstance(meta, KernelDensityEstimate) and meta._fitted:
                        norm = meta.integrated_probability(categories.left[0], categories.right[-1])
                        probabilities = {
                            c: meta.integrated_probability(c.left.to_numpy(), c.right.to_numpy()) / norm
                            for c in categories
                        }
                except ZeroDivisionError:
                    categories = meta.categories
            else:
                categories = meta.categories
        else:
            categories = meta.categories

        hist = Histogram(name=meta.name, categories=categories, nan_freq=meta.nan_freq, probabilities=probabilities)
        hist.dtype = dtype

        return hist

    def plot(self, **kwargs) -> plt.Figure:

        fig = plt.Figure(figsize=(7, 4))
        sns.set_theme(**kwargs)

        plot_data = pd.DataFrame({
            self.name: self.categories,
            'probability': [self.probability(c) for c in self.categories] if self.categories is not None else None
        })
        sns.barplot(data=plot_data, x=self.name, y='probability', ax=fig.gca())
        for tick in fig.gca().get_xticklabels():
            tick.set_rotation(90)

        np.datetime64()
        return fig
