import re
from typing import Optional, List

from dataclasses import dataclass
import numpy as np
import pandas as pd
from faker import Faker

from .categorical import CategoricalMeta
from .value_meta import ValueMeta


@dataclass
class CompoundAddressParams:
    address_label: Optional[str] = None
    postcode_regex: Optional[str] = None


class CompoundAddressMeta(ValueMeta):
    def __init__(self, name, postcode_level=0, address_label: str = None, postcode_regex: str = None):
        super().__init__(name=name)

        if postcode_level < 0 or postcode_level > 2:
            raise NotImplementedError

        self.postcode_level = postcode_level
        if address_label is None:
            raise ValueError
        self.address_label: str = address_label
        self.postcode_regex = postcode_regex

        self.postcodes = None
        self.postcode = CategoricalMeta(name=address_label, categories=self.postcodes)
        self.faker = Faker(locale='en_GB')

    def columns(self) -> List[str]:
        columns = [
            self.address_label
        ]
        return [c for c in columns if c is not None]

    def extract(self, df: pd.DataFrame) -> None:
        super().extract(df=df)

        self.postcodes = {}

        postcode_df = pd.DataFrame({self.address_label: df[self.address_label]})
        for n, row in df.iterrows():
            postcode = row[self.address_label]
            m = re.match(self.postcode_regex, postcode)
            if not m:
                postcode_df.loc[n, self.address_label] = None
                continue
            postcode = m.group(1)
            if self.postcode_level == 0:  # 1-2 letters
                index = 2 - postcode[1].isdigit()
            elif self.postcode_level == 1:
                index = postcode.index(' ')
            elif self.postcode_level == 2:
                index = postcode.index(' ') + 2
            else:
                raise ValueError(self.postcode_level)
            postcode_key = postcode[:index]
            postcode_value = postcode[index:]
            if postcode_key not in self.postcodes:
                self.postcodes[postcode_key] = list()
            self.postcodes[postcode_key].append(postcode_value)

            postcode_df.loc[n, self.address_label] = postcode_key

        postcode_df.dropna(inplace=True)
        self.postcode.extract(df=postcode_df)

    def learned_input_columns(self) -> List[str]:
        return [self.address_label]

    def learned_output_columns(self) -> List[str]:
        return [self.address_label]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        for n, row in df.iterrows():
            postcode = row[self.address_label]
            m = re.match(self.postcode_regex, postcode)
            if not m:
                df.loc[n, self.address_label] = None
                continue
            postcode = m.group(1)
            if self.postcode_level == 0:  # 1-2 letters
                index = 2 - postcode[1].isdigit()
            elif self.postcode_level == 1:
                index = postcode.index(' ')
            elif self.postcode_level == 2:
                index = postcode.index(' ') + 2
            else:
                raise ValueError(self.postcode_level)
            postcode_key = postcode[:index]
            df.loc[n, self.address_label] = postcode_key

        df.dropna(inplace=True)
        df = self.postcode.preprocess(df=df)

        return super().preprocess(df=df)

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().postprocess(df=df)

        if self.postcodes is None or isinstance(self.postcodes, set):
            raise NotImplementedError
        df = self.postcode.postprocess(df=df)
        postcode = df[self.address_label].astype(dtype='str')
        for postcode_key, postcode_values in self.postcodes.items():
            mask = (postcode == postcode_key)
            postcode[mask] += np.random.choice(a=postcode_values, size=mask.sum())

        def expand_address(p):
            address_parts = self.faker.address().split('\n')
            address_parts = address_parts[:-1]
            address_parts.append(p)
            return ', '.join(address_parts)

        df[self.address_label] = postcode.apply(expand_address)
        return df
