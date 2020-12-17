from typing import Generic, Optional, Dict, Any, cast, MutableSequence
from math import ceil

import numpy as np
import pandas as pd

from ..base import DiscreteModel
from ..base.value_meta import NType, Nominal, Ordinal, Affine, SType
from ..exceptions import MetaNotExtractedError, ModelNotFittedError, ExtractionError


class Histogram(DiscreteModel[NType], Generic[NType]):
    class_name = "Histogram"
    MAX_BINS = 20

    def __init__(
            self, name: str, categories: Optional[MutableSequence[NType]] = None, nan_freq: Optional[float] = None,
            probabilities: Optional[Dict[NType, float]] = None, bin_width: Optional[SType] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)  # type: ignore
        self.probabilities: Optional[Dict[NType, float]] = probabilities
        self.bin_width = bin_width

    def fit(self: 'Histogram[NType]', df: pd.DataFrame) -> 'Histogram[NType]':
        super().fit(df=df)
        if self.bin_width is not None:
            value_counts = pd.value_counts(
                ubin_values(df[self.name].astype(self.dtype), self.categories, self.bin_width), dropna=False, sort=False
            )
        else:
            value_counts = df[self.name].value_counts(normalize=True, dropna=False, sort=False)
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
            "probabilities": self.probabilities,
            "precision": self.bin_width
        })
        return d

    @classmethod
    def from_meta(cls, meta: Nominal[NType]) -> 'Histogram[NType]':

        num_categories = len(meta.categories) if meta.categories is not None else 0

        if num_categories <= cls.MAX_BINS:
            hist = Histogram(name=meta.name, categories=meta.categories, nan_freq=meta.nan_freq)
        elif isinstance(meta, Affine) and meta.max is not None and meta.min is not None:
            rng = meta.max - meta.min

            if rng/cls.MAX_BINS > meta.unit_meta.precision:
                precision = rng//cls.MAX_BINS
                num_bins = cls.MAX_BINS
            else:
                precision = meta.unit_meta.precision
                num_bins = ceil(rng/meta.unit_meta.precision)

            categories = [meta.min + i*precision for i in range(num_bins)]

            hist = Histogram(name=meta.name, categories=categories, nan_freq=meta.nan_freq, precision=precision)
        elif isinstance(meta, Ordinal):
            # TODO
            hist = Histogram(name=meta.name, categories=meta.categories[:cls.MAX_BINS], nan_freq=meta.nan_freq)
        else:
            hist = Histogram(name=meta.name, categories=meta.categories, nan_freq=meta.nan_freq)

        if hist is None:
            raise ExtractionError

        hist.dtype = meta.dtype

        return hist


def bin_values(x, categories, binwidth):
    bins = categories[np.argmin(np.abs(x.reshape(-1, 1) - np.reshape(categories, (1,-1))), -1)]
    dist = x.reshape(-1, 1) - np.reshape(categories, (1,-1))
    not_nan = np.any((dist < binwidth) & (dist >= (binwidth*0)), axis=1)
    return np.where(not_nan, bins, None)


ubin_values = np.vectorize(bin_values, otypes=[object], excluded=(2,), signature='(n),(m)->(n)')
