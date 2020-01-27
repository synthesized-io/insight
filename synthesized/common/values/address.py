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
    def __init__(self, postcode, county, city, district, street, house_number, flat, house_name):
        self.postcode = postcode
        self.county = county
        self.city = city
        self.district = district
        self.street = street
        self.house_number = house_number
        self.flat = flat
        self.house_name = house_name

    def __repr__(self):
        return "<AddressRecord {} {} {} {} {} {} {} {}>".format(
            self.postcode,
            self.county,
            self.city,
            self.district,
            self.street,
            self.house_number,
            self.flat,
            self.house_name
        )


class AddressValue(Value):
    postcode_regex = re.compile(r'^[A-Za-z]{1,2}[0-9]+[A-Za-z]? *[0-9]+[A-Za-z]{2}$')

    def __init__(self, name, categorical_kwargs: dict, postcode_level=0, postcode_label=None, county_label=None,
                 city_label=None, district_label=None, street_label=None, house_number_label=None, flat_label=None,
                 house_name_label=None,
                 fake=True):
        super().__init__(name=name)

        if postcode_level < 0 or postcode_level > 2:
            raise NotImplementedError

        self.postcode_level = postcode_level
        self.county_label = county_label
        self.postcode_label = postcode_label
        self.city_label = city_label
        self.district_label = district_label
        self.street_label = street_label
        self.house_number_label = house_number_label
        self.flat_label = flat_label
        self.house_name_label = house_name_label
        self.fake = fake
        self.fkr = faker.Faker(locale='en_GB')

        self.postcodes: Dict[str, List[AddressRecord]] = {}

        assert postcode_label is not None

        if self.fake:
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
        if self.fake:
            return
        for n, row in df.iterrows():
            postcode = row[self.postcode_label]
            postcode_key = self._get_postcode_key(postcode)

            county = row[self.county_label] if self.county_label else None
            city = row[self.city_label] if self.city_label else None
            district = row[self.district_label] if self.district_label else None
            street = row[self.street_label] if self.street_label else None
            house_number = row[self.house_number_label] if self.house_number_label else None
            flat = row[self.flat_label] if self.flat_label else None
            house_name = row[self.house_name_label] if self.house_name_label else None

            if postcode_key not in self.postcodes:
                self.postcodes[postcode_key] = []
            self.postcodes[postcode_key].append(AddressRecord(
                postcode=postcode,
                county=county,
                city=city,
                district=district,
                street=street,
                house_number=house_number,
                flat=flat,
                house_name=house_name)
            )

        # convert list to ndarray for better performance
        for key, postcode in self.postcodes.items():
            self.postcodes[key] = np.array(self.postcodes[key])

        if self.postcode is not None:
            postcode_data = pd.DataFrame({self.postcode_label: list(self.postcodes.keys())})
            self.postcode.extract(df=postcode_data)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.fake:
            return super().preprocess(df=df)
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
        if self.fake:
            if self.postcode_label is not None:
                df[self.postcode_label] = [self.fkr.postcode() for _ in range(len(df))]

            if self.county_label is not None:
                df[self.county_label] = None

            if self.city_label is not None:
                df[self.city_label] = [self.fkr.city() for _ in range(len(df))]

            if self.district_label is not None:
                df[self.district_label] = None

            if self.street_label is not None:
                df[self.street_label] = [self.fkr.street_name() for _ in range(len(df))]

            if self.house_number_label is not None:
                df[self.house_number_label] = [self.fkr.building_number() for _ in range(len(df))]

            if self.flat_label is not None:
                df[self.flat_label] = [self.fkr.secondary_address() for _ in range(len(df))]

            if self.house_name_label is not None:
                df[self.house_name_label] = None
        else:
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

            if self.county_label is not None:
                df[self.county_label] = list(map(lambda a: a.county, addresses))

            if self.city_label:
                df[self.city_label] = list(map(lambda a: a.city, addresses))

            if self.district_label:
                df[self.district_label] = list(map(lambda a: a.district, addresses))

            if self.street_label:
                df[self.street_label] = list(map(lambda a: a.street, addresses))

            if self.house_number_label is not None:
                df[self.house_number_label] = list(map(lambda a: a.house_number, addresses))

            if self.flat_label:
                df[self.flat_label] = list(map(lambda a: a.flat, addresses))

            if self.house_name_label:
                df[self.house_name_label] = list(map(lambda a: a.house_name, addresses))

        return df

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
