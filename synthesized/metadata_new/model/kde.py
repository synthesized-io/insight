from typing import Generic, Optional, Dict, Any, cast, Sequence

import pandas as pd

from ..base import ContinuousModel
from ..base.value_meta import AType
from ..exceptions import MetaNotExtractedError, ModelNotFittedError


class KernelDensityEstimate(ContinuousModel[AType], Generic[AType]):

    def __init__(
            self, name: str, categories: Optional[Sequence[AType]] = None, nan_freq: Optional[float] = None,
            probabilities: Optional[Dict[AType, float]] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq)  # type: ignore
        self.probabilities: Optional[Dict[AType, float]] = probabilities

    def fit(self, df: pd.DataFrame):
        raise NotImplementedError

    def sample(self, n: int) -> pd.DataFrame:
        raise NotImplementedError

    def probability(self, x: Any) -> float:
        if not self._extracted:
            raise MetaNotExtractedError
        if not self._fitted:
            raise ModelNotFittedError

        self.probabilities = cast(Dict[AType, float], self.probabilities)

        if x is None:
            return cast(float, self.nan_freq)

        x = cast(AType, x)

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
