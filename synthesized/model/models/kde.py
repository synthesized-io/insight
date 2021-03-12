from typing import Any, Dict, Generic, Optional, Sequence, Type, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

from ..base import ContinuousModel
from ..exceptions import ModelNotFittedError
from ...metadata_new import Affine, AType, MetaNotExtractedError, Scale


class KernelDensityEstimate(ContinuousModel[Affine[AType], AType], Generic[AType]):

    def __init__(self, meta: Affine[AType]):
        super().__init__(meta=meta)  # type: ignore
        self._kernel: Optional[gaussian_kde] = None

    @property
    def min(self) -> AType:
        if self._meta.min is None:
            raise MetaNotExtractedError
        else:
            return self._meta.min

    @property
    def max(self) -> AType:
        if self._meta.max is None:
            raise MetaNotExtractedError
        else:
            return self._meta.max

    def fit(self, df: pd.DataFrame):
        super().fit(df)
        c = self.min if self.min is not None else np.array(0, dtype=self._meta.dtype)

        self._kernel = gaussian_kde(
            (df[self.name].dropna().values.astype(self._meta.dtype) - c).astype(self.kde_dtype),
            bw_method='silverman'
        )
        return self

    @property
    def kde_dtype(self) -> str:
        dtype_map = {
            'm8[ns]': 'i8'
        }
        if self._meta.unit_meta.dtype is None:
            raise ValueError
        return dtype_map.get(self._meta.unit_meta.dtype, self._meta.unit_meta.dtype)

    def sample(self, n: int, produce_nans: bool = False, conditions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self._fitted:
            raise ModelNotFittedError

        df = pd.DataFrame({self.name: np.squeeze(cast(gaussian_kde, self._kernel).resample(size=n)).astype(
            self._meta.unit_meta.dtype) + self.min})

        if produce_nans:
            df[self.name] = self.add_nans(df[self.name], self.nan_freq)

        return df

    def probability(self, x: Any) -> float:
        if not self._fitted:
            raise ModelNotFittedError

        if x is None:
            return cast(float, self.nan_freq)

        c = self.min if self.min is not None else np.array(0, dtype=self._meta.dtype)
        x = (x - c).astype(self.kde_dtype)

        if self.kde_dtype == 'i8':
            half_win = self._meta.unit_meta.precision.astype(self.kde_dtype) / 2
            if not np.isscalar(x):
                prob = np.array([cast(gaussian_kde, self._kernel).integrate_box_1d(low=y - half_win, high=y + half_win) for y in x])
            else:
                prob = cast(gaussian_kde, self._kernel).integrate_box_1d(low=(x - half_win), high=x + half_win)
        else:
            prob = cast(gaussian_kde, self._kernel)(x)

        return prob

    def integrate(self, low: Any, high: Any) -> float:
        if not self._fitted:
            raise ModelNotFittedError

        c = self.min if self.min is not None else np.array(0, dtype=self._meta.dtype)
        low = np.array(low, dtype=self._meta.dtype)
        high = np.array(high, dtype=self._meta.dtype)

        low = np.array(low - c, dtype=self.kde_dtype).item()
        high = np.array(high - c, dtype=self.kde_dtype).item()

        prob = cast(gaussian_kde, self._kernel).integrate_box_1d(low=low, high=high)

        return prob

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        return d

    @property
    def unit_meta(self) -> 'Scale[Any]':
        return self._meta._unit_meta

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
