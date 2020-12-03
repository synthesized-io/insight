from typing import Generic, Optional, Dict, Any, cast

import pandas as pd

from ..base import DiscreteModel, Domain
from ..base.value_meta import NType
from ..exceptions import MetaNotExtractedError, ModelNotFittedError


class Histogram(DiscreteModel[NType], Generic[NType]):
    class_name = "DiscreteModel"

    def __init__(
            self, name: str, domain: Optional[Domain[NType]] = None, nan_freq: Optional[float] = None,
            probabilities: Optional[Dict[NType, float]] = None
    ):
        super().__init__(name=name, domain=domain, nan_freq=nan_freq)  # type: ignore
        self.probabilities: Optional[Dict[NType, float]] = probabilities

    def fit(self, df: pd.DataFrame) -> 'Histogram[NType]':
        super().fit(df=df)
        value_counts = df[self.name].value_counts(normalize=True, dropna=False, sort=False)
        self.probabilities = {cat: value_counts[cat] for cat in value_counts}
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
