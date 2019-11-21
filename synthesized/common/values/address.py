import re
from typing import List, Dict

import faker
import numpy as np
import pandas as pd
import tensorflow as tf

from .categorical import CategoricalValue
from .value import Value
from ..module import tensorflow_name_scoped


class AddressRecord:
    def __init__(self, city, postcode, street, house_number):
        self.city = city
        self.postcode = postcode
        self.street = street
        self.house_number = house_number

    def __repr__(self):
        return "<AddressRecord {}, {}, {}, {}>".format(self.city, self.postcode, self.street, self.house_number)


class AddressValue(Value):
    postcode_regex = re.compile(r'^[A-Za-z]{1,2}[0-9]+[A-Za-z]? *[0-9]+[A-Za-z]{2}$')

    def __init__(self, name, categorical_kwargs: dict, postcode_level=0, postcode_label=None,
                 city_label=None, street_label=None, house_number_label=None, fake=False):
        super().__init__(name=name)

        if postcode_level < 0 or postcode_level > 2:
            raise NotImplementedError

        self.postcode_level = postcode_level
        self.postcode_label = postcode_label
        self.city_label = city_label
        self.street_label = street_label
        self.house_number_label = house_number_label
        self.fake = fake
        self.fkr = faker.Faker(locale='en_GB')

        self.postcodes: Dict[str, List[AddressRecord]] = {}

        if postcode_label is None:
            self.postcode = None
        else:
            self.postcode = self.add_module(
                module=CategoricalValue, name=postcode_label,
                **categorical_kwargs
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
        for n, row in df.iterrows():
            postcode = row[self.postcode_label]
            postcode_key = self._get_postcode_key(postcode)

            city = row[self.city_label] if self.city_label else None
            street = row[self.street_label] if self.street_label else None
            house_number = row[self.house_number_label] if self.house_number_label else None

            if postcode_key not in self.postcodes:
                self.postcodes[postcode_key] = []
            self.postcodes[postcode_key].append(AddressRecord(city, postcode, street, house_number))

        # convert list to ndarray for better performance
        for key, postcode in self.postcodes.items():
            self.postcodes[key] = np.array(self.postcodes[key])

        if self.postcode is not None:
            postcode_data = pd.DataFrame({self.postcode_label: list(self.postcodes.keys())})
            self.postcode.extract(df=postcode_data)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        postcodes = []
        for n, row in df.iterrows():
            postcode = row[self.postcode_label]
            postcode_key = self._get_postcode_key(postcode)
            postcodes.append(postcode_key)

        df[self.postcode_label] = postcodes

        if self.postcode is not None:
            df = self.postcode.preprocess(df=df)

        return super().preprocess(df=df)

    def _get_postcode_key(self, postcode: str):
        if not AddressValue.postcode_regex.match(postcode):
            raise ValueError(postcode)
        if self.postcode_level == 0:  # 1-2 letters
            index = 2 - postcode[1].isdigit()
        elif self.postcode_level == 1:
            index = postcode.index(' ')
        elif self.postcode_level == 2:
            index = postcode.index(' ') + 2
        else:
            raise ValueError(self.postcode_level)
        return postcode[:index]

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().postprocess(df=df)

        if self.postcodes is None or isinstance(self.postcodes, set):
            raise NotImplementedError
        if self.postcode is None:
            postcode = np.random.choice(a=list(self.postcodes), size=len(df))
        else:
            df = self.postcode.postprocess(df=df)
            postcode = df[self.postcode_label].astype(dtype='str').to_numpy()

        def sample_address(postcode_key):
            return np.random.choice(self.postcodes[postcode_key])

        addresses = np.vectorize(sample_address)(postcode)

        df[self.postcode_label] = list(map(lambda a: a.postcode, addresses))

        if self.city_label:
            df[self.city_label] = list(map(lambda a: a.city, addresses))

        if self.street_label:
            df[self.street_label] = list(map(lambda a: a.street, addresses))

        if self.house_number_label:
            df[self.house_number_label] = list(map(lambda a: a.house_number, addresses))

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
