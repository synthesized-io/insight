import re
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf

from .value import Value
from .categorical import CategoricalValue
from ..module import tensorflow_name_scoped


# TODO: postcodes needs one-time scan to collect
# street etc optional


class AddressValue(Value):
    postcode_regex = re.compile(r'^[A-Za-z]{1,2}[0-9]+[A-Za-z]? [0-9]+[A-Za-z]{2}$')

    def __init__(self, name, postcode_level=0, postcode_label=None, capacity=None, city_label=None, street_label=None,
                 postcodes=None):
        super().__init__(name=name)

        if postcode_level < 0 or postcode_level > 2:
            raise NotImplementedError

        self.postcode_level = postcode_level
        self.postcode_label = postcode_label
        self.city_label = city_label
        self.street_label = street_label

        if postcodes is None:
            self.postcodes = None
        else:
            self.postcodes = sorted(postcodes)

        self.streets = {}
        self.cities = {}

        if postcode_label is None:
            self.postcode = None
        else:
            self.postcode = self.add_module(
                module=CategoricalValue, name=postcode_label, categories=self.postcodes,
                capacity=capacity
            )

    def learned_input_columns(self) -> List[str]:
        if self.postcode is None:
            return super().learned_input_columns()
        else:
            return self.postcode.learned_input_columns()

    def learned_output_columns(self) -> List[str]:
        if self.postcode is None:
            return super().learned_output_columns()
        else:
            return self.postcode.learned_output_columns()

    def learned_input_size(self) -> int:
        if self.postcode is None:
            return super().learned_input_size()
        else:
            return self.postcode.learned_input_size()

    def learned_output_size(self) -> int:
        if self.postcode is None:
            return super().learned_output_size()
        else:
            return self.postcode.learned_output_size()

    def extract(self, df: pd.DataFrame) -> None:
        super().extract(df=df)

        if self.postcodes is None:
            self.postcodes = dict()
            fixed = False
        else:
            self.postcodes = {postcode: list() for postcode in self.postcodes}
            fixed = True

        keys = []
        for n, row in df.iterrows():
            postcode = row[self.postcode_label]
            if not self.__class__.postcode_regex.match(postcode):
                raise ValueError(postcode)
            if self.postcode_level == 0:  # 1-2 letters
                index = 2 - postcode[1].isdigit()
            elif self.postcode_level == 1:
                index = postcode.index(' ')
            elif self.postcode_level == 2:
                index = postcode.index(' ') + 2
            postcode_key = postcode[:index]
            postcode_value = postcode[index:]
            if postcode_key not in self.postcodes:
                if fixed:
                    raise NotImplementedError
                else:
                    self.postcodes[postcode_key] = []
            self.postcodes[postcode_key].append(postcode_value)

            if self.street_label:
                self.streets[postcode_key] = row[self.street_label]

            if self.city_label:
                self.cities[postcode_key] = row[self.city_label]

            keys.append(postcode_key)

        # convert list to ndarray for better performance
        for key, postcode in self.postcodes.items():
            self.postcodes[key] = np.array(self.postcodes[key])

        if self.postcode is not None:
            postcode_data = pd.DataFrame({self.postcode_label: keys})
            self.postcode.extract(df=postcode_data)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        postcodes = []
        for n, row in df.iterrows():
            postcode = row[self.postcode_label]
            if not self.__class__.postcode_regex.match(postcode):
                raise ValueError(postcode)
            if self.postcode_level == 0:  # 1-2 letters
                index = 2 - postcode[1].isdigit()
            elif self.postcode_level == 1:
                index = postcode.index(' ')
            elif self.postcode_level == 2:
                index = postcode.index(' ') + 2
            postcode_key = postcode[:index]
            postcodes.append(postcode_key)

        df[self.postcode_label] = postcodes

        if self.postcode is not None:
            df = self.postcode.preprocess(df=df)

        return super().preprocess(df=df)

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().postprocess(df=df)

        if self.postcodes is None or isinstance(self.postcodes, set):
            raise NotImplementedError
        if self.postcode is None:
            postcode = pd.Series(data=np.random.choice(a=list(self.postcodes), size=len(df)),
                                 name=self.postcode_label)
        else:
            df = self.postcode.postprocess(data=df)
            postcode = df[self.postcode_label].astype(dtype='str')

        def expand_postcode(key):
            return key + np.random.choice(self.postcodes[key])

        def lookup_city(key):
            return self.cities[key]

        def lookup_street(key):
            return self.streets[key]

        df[self.postcode_label] = postcode.apply(expand_postcode)

        if self.city_label:
            df[self.city_label] = postcode.apply(lookup_city)

        if self.street_label:
            df[self.street_label] = postcode.apply(lookup_street)

        return df

    @tensorflow_name_scoped
    def input_tensors(self) -> List[tf.Tensor]:
        if self.postcode is None:
            return super().input_tensors()
        else:
            return self.postcode.input_tensors()

    @tensorflow_name_scoped
    def unify_inputs(self, xs: List[tf.Tensor]) -> tf.Tensor:
        if self.postcode is None:
            return super().unify_inputs(xs=xs)
        else:
            return self.postcode.unify_inputs(xs=xs)

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor) -> List[tf.Tensor]:
        if self.postcode is None:
            return super().output_tensors(y=y)
        else:
            return self.postcode.output_tensors(y=y)

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, xs: List[tf.Tensor]) -> tf.Tensor:
        if self.postcode is None:
            return super().loss(y=y, xs=xs)
        else:
            return self.postcode.loss(y=y, xs=xs)

    @tensorflow_name_scoped
    def distribution_loss(self, ys: List[tf.Tensor]) -> tf.Tensor:
        if self.postcode is None:
            return super().distribution_loss(ys=ys)
        else:
            return self.postcode.distribution_loss(ys=ys)
