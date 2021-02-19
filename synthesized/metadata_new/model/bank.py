from typing import Optional, Sequence

import numpy as np
import pandas as pd
from faker import Faker

from ..base import DiscreteModel
from ..base.value_meta import NType
from ..exceptions import ModelNotFittedError
from ..value import Bank
from ...config import BankLabels


class BankModel(Bank, DiscreteModel[str]):

    def __init__(self, name: str, categories: Optional[Sequence[str]] = None, nan_freq: Optional[float] = None,
                 labels: BankLabels = BankLabels()):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq, labels=labels)

        if nan_freq is not None:
            self._fitted = True

    def sample(self, n: int, produce_nans: bool = False) -> pd.DataFrame:
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
            for c in self._params.values():
                if c is not None:
                    is_nan = np.random.binomial(1, p=self.nan_freq, size=n) == 1
                    df.loc[is_nan, c] = np.nan

        return df

    @classmethod
    def from_meta(cls, meta: 'Bank') -> 'BankModel':
        assert isinstance(meta, Bank)
        return cls(name=meta.name, nan_freq=meta.nan_freq, labels=meta.labels)
