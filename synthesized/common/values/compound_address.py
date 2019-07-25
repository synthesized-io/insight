import numpy as np
import pandas as pd
import re
from typing import List

from faker import Faker
import tensorflow as tf

from .value import Value
from .categorical import CategoricalValue
from ..module import tensorflow_name_scoped


class CompoundAddressValue(Value):
    def __init__(self, name, postcode_level=0, address_label=None, postcode_regex=None, capacity=None):
        super().__init__(name=name)

        if postcode_level < 0 or postcode_level > 2:
            raise NotImplementedError

        self.postcode_level = postcode_level
        self.address_label = address_label
        self.postcode_regex = postcode_regex

        self.postcodes = None
        self.postcode = self.add_module(
            module=CategoricalValue, name=address_label, categories=self.postcodes,
            capacity=capacity
        )
        self.faker = Faker(locale='en_GB')

    def learned_input_columns(self) -> List[str]:
        return self.postcode.learned_input_columns()

    def learned_output_columns(self) -> List[str]:
        return self.postcode.learned_output_columns()

    def learned_input_size(self) -> int:
        return self.postcode.learned_input_size()

    def learned_output_size(self) -> int:
        return self.postcode.learned_output_size()

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

    @tensorflow_name_scoped
    def input_tensors(self) -> List[tf.Tensor]:
        return self.postcode.input_tensors()

    @tensorflow_name_scoped
    def unify_inputs(self, xs: List[tf.Tensor]) -> tf.Tensor:
        return self.postcode.unify_inputs(xs=xs)

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor) -> List[tf.Tensor]:
        return self.postcode.output_tensors(y=y)

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, xs: List[tf.Tensor]) -> tf.Tensor:
        return self.postcode.loss(y=y, xs=xs)

    @tensorflow_name_scoped
    def distribution_loss(self, ys: List[tf.Tensor]) -> tf.Tensor:
        return self.postcode.distribution_loss(ys=ys)