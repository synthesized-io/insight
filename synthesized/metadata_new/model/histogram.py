from typing import Generic, Optional, Dict, Any, cast, Sequence, overload, Union, Type

import pandas as pd

from ..base import DiscreteModel
from ..base.value_meta import NType, Nominal, Ordinal, Affine, AType
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
        pass

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
    @overload
    def from_meta(cls: Type['Histogram'], meta: Nominal[AType], max_bins: int = 20) -> 'Union[Histogram[AType], Histogram[pd.IntervalDtype[AType]]]':
        ...

    @classmethod
    @overload
    def from_meta(cls: Type['Histogram'], meta: Nominal[NType], max_bins: int = 20) -> 'Histogram[NType]':
        ...

    @classmethod
    def from_meta(cls: Type['Histogram'], meta: Nominal[Any], max_bins: int = 20) -> 'Histogram[Any]':

        num_categories = len(meta.categories) if meta.categories is not None else 0
        dtype = meta.dtype

        if num_categories <= max_bins:
            hist = Histogram(name=meta.name, categories=meta.categories, nan_freq=meta.nan_freq)
        elif isinstance(meta, Affine) and meta.max is not None and meta.min is not None:
            rng = meta.max - meta.min

            if (rng / max_bins) > meta.unit_meta.precision:
                bin_width = rng // max_bins
            else:
                bin_width = meta.unit_meta.precision

            categories = pd.interval_range(meta.min, meta.max, freq=bin_width.item(), closed='left')
            dtype = str(categories.dtype)  # TODO: find away to handle 'interval[M8[D]]' instead of 'interval[M8[ns]]'

            hist = Histogram(name=meta.name, categories=categories, nan_freq=meta.nan_freq)
        elif isinstance(meta, Ordinal):
            categories = meta.categories[:max_bins] if meta.categories is not None else None
            hist = Histogram(name=meta.name, categories=categories, nan_freq=meta.nan_freq)
        else:
            categories = meta.categories[:max_bins] if meta.categories is not None else None
            hist = Histogram(name=meta.name, categories=categories, nan_freq=meta.nan_freq)

        if hist is None:
            raise ExtractionError

        hist.dtype = dtype

        return hist
