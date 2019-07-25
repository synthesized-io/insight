from typing import List

import faker
import numpy as np
import pandas as pd
import tensorflow as tf

from .categorical import CategoricalValue
from .value import Value
from ..module import tensorflow_name_scoped


class PersonValue(Value):

    def __init__(self, name, title_label=None, gender_label=None, name_label=None, firstname_label=None,
                 lastname_label=None, email_label=None, capacity=None, dict_cache_size=10000):
        super().__init__(name=name)

        self.title_label = title_label
        self.gender_label = gender_label
        self.name_label = name_label
        self.firstname_label = firstname_label
        self.lastname_label = lastname_label
        self.email_label = email_label
        # Assume the gender are always encoded like M or F or U(???)
        self.title_mapping = {'M': 'Mr', 'F': 'Mrs', 'U': 'female'}

        fkr = faker.Faker(locale='en_GB')
        self.male_first_name_cache = np.array(list({fkr.first_name_male() for _ in range(dict_cache_size)}))
        self.female_first_name_cache = np.array(list({fkr.first_name_female() for _ in range(dict_cache_size)}))
        self.last_name_cache = np.array(list({fkr.last_name() for _ in range(dict_cache_size)}))

        if gender_label is None:
            self.gender = None
        else:
            self.gender = self.add_module(
                module=CategoricalValue, name=gender_label, capacity=capacity
            )

    def learned_input_columns(self) -> List[str]:
        if self.gender is None:
            return super().learned_input_columns()
        else:
            return self.gender.learned_input_columns()

    def learned_output_columns(self) -> List[str]:
        if self.gender is None:
            return super().learned_output_columns()
        else:
            return self.gender.learned_output_columns()

    def learned_input_size(self) -> int:
        if self.gender is None:
            return super().learned_input_size()
        else:
            return self.gender.learned_input_size()

    def learned_output_size(self) -> int:
        if self.gender is None:
            return super().learned_output_size()
        else:
            return self.gender.learned_output_size()

    def extract(self, df: pd.DataFrame) -> None:
        super().extract(df=df)
        if self.gender is not None:
            self.gender.extract(df=df)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.gender is not None:
            df = self.gender.preprocess(df=df)
        return super().preprocess(df=df)

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().postprocess(df=df)

        if self.gender is None:
            gender = np.random.choice(a=['female', 'male'], size=len(df))
        else:
            df = self.gender.postprocess(data=df)
            gender = df[self.gender_label]

        def get_first_name(g):
            if g == 'M':
                return np.random.choice(self.male_first_name_cache)
            else:
                return np.random.choice(self.female_first_name_cache)

        title = gender.astype(dtype=str).apply(func=self.title_mapping.__getitem__)
        firstname = gender.astype(dtype=str).apply(func=get_first_name)
        lastname = pd.Series(data=np.random.choice(self.last_name_cache, size=len(df)))

        if self.title_label is not None:
            df.loc[:, self.title_label] = title
        if self.name_label is not None:
            df.loc[:, self.name_label] = firstname.str.cat(others=lastname, sep=' ')
        if self.firstname_label is not None:
            df.loc[:, self.firstname_label] = firstname
        if self.lastname_label is not None:
            df.loc[:, self.lastname_label] = lastname
        if self.email_label is not None:
            # https://email-verify.my-addr.com/list-of-most-popular-email-domains.php
            # we don't want clashes with real emails
            # domain = np.random.choice(a=['gmail.com', 'yahoo.com', 'hotmail.com'], size=len(data))
            df.loc[:, self.email_label] = firstname.str.lower() \
                .str.cat(others=lastname.str.lower(), sep='.')
            df.loc[:, self.email_label] += '@example.com'
        return df

    @tensorflow_name_scoped
    def input_tensors(self) -> List[tf.Tensor]:
        if self.gender is None:
            return super().input_tensors()
        else:
            return self.gender.input_tensors()

    @tensorflow_name_scoped
    def unify_inputs(self, xs: List[tf.Tensor]) -> tf.Tensor:
        if self.gender is None:
            return super().unify_inputs(xs=xs)
        else:
            return self.gender.unify_inputs(xs=xs)

    @tensorflow_name_scoped
    def output_tensors(self, y: tf.Tensor) -> List[tf.Tensor]:
        if self.gender is None:
            return super().output_tensors(y=y)
        else:
            return self.gender.output_tensors(y=y)

    @tensorflow_name_scoped
    def loss(self, y: tf.Tensor, xs: List[tf.Tensor]) -> tf.Tensor:
        if self.gender is None:
            return super().loss(y=y, xs=xs)
        else:
            return self.gender.loss(y=y, xs=xs)

    @tensorflow_name_scoped
    def distribution_loss(self, ys: List[tf.Tensor]) -> tf.Tensor:
        if self.gender is None:
            return super().distribution_loss(ys=ys)
        else:
            return self.gender.distribution_loss(ys=ys)