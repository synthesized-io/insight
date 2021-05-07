from typing import Dict, Generic, Optional, cast, Any

import numpy as np
import pandas as pd

from ..base import ContinuousModel
from ..exceptions import ModelNotFittedError
from ...metadata import Affine, AType, Scale


class EnumerationModel(ContinuousModel[Affine[AType], AType], Generic[AType]):
    def __init__(self, meta: Affine):
        super().__init__(meta=meta)

    @property
    def min(self) -> Optional[AType]:
        return self._meta.min

    @property
    def max(self) -> Optional[AType]:
        return self._meta.max

    @property
    def diff(self) -> Optional[AType]:
        return self.unit_meta.categories[0]

    @property
    def unit_meta(self) -> Scale[Any]:
        return self._meta.unit_meta

    def fit(self, df: pd.DataFrame):
        super().fit(df)
        return self

    def sample(self, n: int, produce_nans: bool = False, conditions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self._fitted:
            raise ModelNotFittedError

        min_idx = cast(int, self.min)
        idx_diff = cast(int, self.diff)
        end_idx = min_idx + (n * idx_diff)
        enumerated_col = np.arange(start=min_idx, stop=end_idx, step=idx_diff)
        df = pd.DataFrame({self.name: enumerated_col})

        return df

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> 'EnumerationModel':
        meta_dict = cast(Dict[str, object], d["meta"])
        meta = Affine[AType].from_dict(meta_dict)
        model = cls(meta=meta)
        model._fitted = cast(bool, d["fitted"])
        return model
