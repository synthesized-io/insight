from typing import Dict, Optional, Sequence, cast

import numpy as np
import pandas as pd
from faker import Faker

from ..base import DiscreteModel
from ..exceptions import ModelNotFittedError
from ...config import BankLabels
from ...metadata.value import Bank


class BankModel(DiscreteModel[Bank, str]):

    def __init__(self, meta: Bank):
        super().__init__(meta=meta)

        if self._meta.nan_freq is not None:
            self._fitted = True

    @property
    def categories(self) -> Sequence[str]:
        if self._meta.categories is None:
            raise ModelNotFittedError

        return self._meta.categories

    @property
    def labels(self) -> BankLabels:
        return self._meta.labels

    @property
    def params(self) -> Dict[str, str]:
        return self._meta.params

    def sample(self, n: int, produce_nans: bool = False, conditions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self._fitted:
            raise ModelNotFittedError

        fkr = Faker(locale='en_GB')
        ibans = [fkr.iban() for _ in range(n)]

        df = pd.DataFrame([[], ] * n)

        if self.labels.bic_label is not None:
            df[self.labels.bic_label] = [str(iban[4:8]) for iban in ibans]

        if self.labels.sort_code_label is not None:
            df[self.labels.sort_code_label] = [str(iban[8:14]) for iban in ibans]

        if self.labels.account_label is not None:
            df[self.labels.account_label] = [str(iban[14:]) for iban in ibans]

        if produce_nans and (self.nan_freq is not None and self.nan_freq > 0):
            for c in self.params.values():
                if c is not None:
                    is_nan = np.random.binomial(1, p=self.nan_freq, size=n) == 1
                    df.loc[is_nan, c] = np.nan

        return df

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> 'BankModel':
        meta_dict = cast(Dict[str, object], d["meta"])
        meta = Bank.from_dict(meta_dict)
        model = cls(meta=meta)
        model._fitted = cast(bool, d["fitted"])

        return model
