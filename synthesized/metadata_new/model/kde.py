from typing import Any, Dict, Generic, Optional, Sequence, Type, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

from ..base import ContinuousModel
from ..base.value_meta import Affine, AType, Scale
from ..exceptions import MetaNotExtractedError, ModelNotFittedError


class KernelDensityEstimate(ContinuousModel[AType], Generic[AType]):

    def __init__(
            self, name: str, categories: Optional[Sequence[AType]] = None, nan_freq: Optional[float] = None,
            min: Optional[AType] = None, max: Optional[AType] = None, unit_meta=None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq, min=min, max=max)  # type: ignore
        self._kernel: Optional[gaussian_kde] = None
        self._unit_meta = unit_meta
        self._extracted = True

    def fit(self, df: pd.DataFrame):
        super().fit(df)
        c = self.min if self.min is not None else np.array(0, dtype=self.dtype)

        self._kernel = gaussian_kde(
            (df[self.name].dropna().values.astype(self.dtype) - c).astype(self.kde_dtype),
            bw_method='silverman'
        )
        return self

    @property
    def kde_dtype(self) -> str:
        dtype_map = {
            'm8[D]': 'i8'
        }
        if self.unit_meta.dtype is None:
            raise ValueError
        return dtype_map.get(self.unit_meta.dtype, self.unit_meta.dtype)

    def sample(self, n: int, produce_nans: bool = False) -> pd.DataFrame:
        if not self._fitted:
            raise ModelNotFittedError

        df = pd.DataFrame({self.name: np.squeeze(cast(gaussian_kde, self._kernel).resample(size=n)).astype(
            self.unit_meta.dtype) + self.min})

        if produce_nans:
            df[self.name] = self.add_nans(df[self.name], self.nan_freq)

        return df

    def probability(self, x: Any) -> float:
        if not self._extracted:
            raise MetaNotExtractedError
        if not self._fitted:
            raise ModelNotFittedError

        if x is None:
            return cast(float, self.nan_freq)

        c = self.min if self.min is not None else np.array(0, dtype=self.dtype)
        x = (x - c).astype(self.kde_dtype)

        if self.kde_dtype == 'i8':
            half_win = self.unit_meta.precision.astype(self.kde_dtype) / 2
            if not np.isscalar(x):
                prob = np.array([cast(gaussian_kde, self._kernel).integrate_box_1d(low=y - half_win, high=y + half_win) for y in x])
            else:
                prob = cast(gaussian_kde, self._kernel).integrate_box_1d(low=(x - half_win), high=x + half_win)
        else:
            prob = cast(gaussian_kde, self._kernel)(x)

        return prob

    def integrate(self, low: Any, high: Any) -> float:
        if not self._extracted:
            raise MetaNotExtractedError
        if not self._fitted:
            raise ModelNotFittedError

        c = self.min if self.min is not None else np.array(0, dtype=self.dtype)
        low = np.array(low, dtype=self.dtype)
        high = np.array(high, dtype=self.dtype)

        low = np.array(low - c, dtype=self.kde_dtype).item()
        high = np.array(high - c, dtype=self.kde_dtype).item()

        prob = cast(gaussian_kde, self._kernel).integrate_box_1d(low=low, high=high)

        return prob

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        return d

    @classmethod
    def from_meta(cls: Type['KernelDensityEstimate'], meta: Affine[AType]) -> 'KernelDensityEstimate[AType]':
        kde = cls(
            name=meta.name, categories=meta.categories, nan_freq=meta.nan_freq, min=meta.min, max=meta.max,
            unit_meta=meta.unit_meta
        )
        kde.dtype = meta.dtype

        return kde

    @property
    def unit_meta(self) -> 'Scale[Any]':
        return self._unit_meta

    def plot(self, kde_grid_num=100, **kwargs) -> plt.Figure:
        if self.max is None or self.min is None:
            raise ValueError

        fig = plt.Figure(figsize=(7, 4))
        sns.set_theme(**kwargs)

        domain = np.linspace(
            start=np.array(0, dtype=self.kde_dtype),
            stop=np.array(self.max - self.min, dtype=self.kde_dtype),
            num=kde_grid_num
        ).astype(self.unit_meta.dtype) + self.min

        plot_data = pd.DataFrame({
            self.name: domain,
            'pdf': self.probability(domain)
        })

        if self.kde_dtype == 'i8':
            sns.barplot(data=plot_data, x=self.name, y='pdf', ax=fig.gca())
        else:
            sns.lineplot(data=plot_data, x=self.name, y='pdf', ax=fig.gca())

        for tick in fig.gca().get_xticklabels():
            tick.set_rotation(90)

        return fig
