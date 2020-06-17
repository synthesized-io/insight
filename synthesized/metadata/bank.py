from typing import List, Union

from dataclasses import dataclass
import faker
import pandas as pd

from .value_meta import ValueMeta


@dataclass
class BankParams:
    bic_label: Union[str, List[str], None] = None
    sort_code_label: Union[str, List[str], None] = None
    account_label: Union[str, List[str], None] = None


class BankNumberMeta(ValueMeta):
    def __init__(self, name, bic_label=None, sort_code_label=None, account_label=None):
        super().__init__(name)
        fkr = faker.Faker(locale='en_GB')
        self.fkr = fkr
        self.bic_label = bic_label
        self.sort_code_label = sort_code_label
        self.account_label = account_label

    def extract(self, df: pd.DataFrame) -> None:
        pass

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(labels=[self.sort_code_label, self.account_label], axis=1)
        return super().preprocess(df=df)

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().postprocess(df=df)
        ibans = [self.fkr.iban() for _ in range(len(df))]
        if self.bic_label is not None:
            df.loc[:, self.bic_label] = [iban[4:8] for iban in ibans]
        if self.sort_code_label is not None:
            df.loc[:, self.sort_code_label] = [int(iban[8:14]) for iban in ibans]
        if self.account_label is not None:
            df.loc[:, self.account_label] = [int(iban[14:]) for iban in ibans]
        return df
