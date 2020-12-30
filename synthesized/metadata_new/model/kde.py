from typing import Generic, Optional, Dict, Any, cast, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import seaborn as sns

from ..base import ContinuousModel
from ..base.value_meta import AType, Affine, Scale
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
            (df[self.name].values.astype(self.dtype) - c).astype(self.kde_dtype),
            bw_method='silverman'
        )

    @property
    def kde_dtype(self) -> str:
        dtype_map = {
            'm8[D]': 'i8'
        }
        return dtype_map.get(self.unit_meta.dtype, self.unit_meta.dtype)

    def sample(self, n: int) -> pd.DataFrame:
        if not self._fitted:
            raise ModelNotFittedError
        return pd.DataFrame(
            {self.name: np.squeeze(self._kernel.resample(size=n)).astype(self.unit_meta.dtype) + self.min}
        )

    def probability(self, x: Any) -> float:
        if not self._extracted:
            raise MetaNotExtractedError
        if not self._fitted:
            raise ModelNotFittedError

        if x is None:
            return cast(float, self.nan_freq)

        c = self.min if self.min is not None else np.array(0, dtype=self.dtype)
        x = (x - c).astype(self.kde_dtype)

        prob = self._kernel(x)

        return prob

    def integrated_probability(self, low: Any, high: Any) -> float:
        if not self._extracted:
            raise MetaNotExtractedError
        if not self._fitted:
            raise ModelNotFittedError

        c = self.min if self.min is not None else np.array(0, dtype=self.dtype)
        low = np.array(low, dtype=self.dtype)
        high = np.array(high, dtype=self.dtype)

        low = np.array(low - c, dtype=self.kde_dtype).item()
        high = np.array(high - c, dtype=self.kde_dtype).item()

        prob = self._kernel.integrate_box_1d(low=low, high=high)

        return prob

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        return d

    @classmethod
    def from_meta(cls, meta: Affine[AType]) -> 'KernelDensityEstimate[AType]':
        kde = KernelDensityEstimate(
            name=meta.name, categories=meta.categories, nan_freq=meta.nan_freq, min=meta.min, max=meta.max,
            unit_meta=meta.unit_meta
        )
        kde.dtype = meta.dtype

        return kde

    @property
    def unit_meta(self) -> 'Scale[Any]':
        return self._unit_meta

    def plot(self, kde_grid_num=100, **kwargs) -> plt.Figure:

        fig = plt.Figure(figsize=(7, 4))
        sns.set_theme(**kwargs)

        domain = np.linspace(
            start=np.array(0, dtype=self.kde_dtype),
            stop=np.array(self.max-self.min, dtype=self.kde_dtype),
            num=kde_grid_num
        ).astype(self.unit_meta.dtype) + self.min

        plot_data = pd.DataFrame({
            self.name: domain,
            'pdf': self.probability(domain)
        })
        sns.lineplot(data=plot_data, x=self.name, y='pdf', ax=fig.gca())
        for tick in fig.gca().get_xticklabels():
            tick.set_rotation(90)

        return fig
