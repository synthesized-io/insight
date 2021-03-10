import importlib
import random
import re
import string
from typing import Optional, Sequence

import faker
import numpy as np
import pandas as pd
from faker.providers.person.en import Provider

from .histogram import Histogram
from ..base import DiscreteModel
from ...config import GenderTransformerConfig, PersonLabels, PersonModelConfig
from ...metadata_new.value import Person
from ...util import get_gender_from_df, get_gender_title_from_df


class GenderModel(Histogram[str]):

    def __init__(self, name, nan_freq: Optional[float] = None,
                 gender_label: Optional[str] = None, title_label: Optional[str] = None,
                 config: GenderTransformerConfig = GenderTransformerConfig()):

        self.config = config
        self.gender_label = gender_label
        self.title_label = title_label

        super().__init__(name=name, categories=self.config.genders, nan_freq=nan_freq,
                         probabilities={gender: 1 / len(self.config.genders) for gender in self.config.genders})

    def fit(self, df):
        df_gender = get_gender_from_df(df[[self.gender_label or self.title_label]].copy(), name=self.name,
                                       gender_label=self.gender_label, title_label=self.title_label,
                                       gender_mapping=self.config.gender_mapping,
                                       title_mapping=self.config.title_mapping)
        return super().fit(df_gender)

    def sample(self, n: Optional[int], produce_nans: bool = False, conditions: Optional[pd.DataFrame] = None):
        if n is None and conditions is None:
            raise ValueError("One of 'n' or 'conditions' must be given")
        elif n is None and conditions is not None:
            n = len(conditions)

        assert n is not None
        df = super().sample(n, produce_nans=produce_nans)
        return get_gender_title_from_df(df, name=self.name, gender_label=self.gender_label,
                                        title_label=self.title_label, gender_mapping=self.config.gender_mapping,
                                        title_mapping=self.config.title_mapping)


class PersonModel(Person, DiscreteModel[str]):

    def __init__(self, name, categories: Optional[Sequence[str]] = None, nan_freq: Optional[float] = None,
                 labels: PersonLabels = PersonLabels(), config: PersonModelConfig = PersonModelConfig()):

        columns = [c for c in labels.__dict__.values() if c is not None]
        if all([c is None for c in columns]):
            raise ValueError("At least one of labels must be given")
        if name in columns:
            raise ValueError("Value of 'name' can't be equal to any other label.")
        if len(columns) > len(np.unique(columns)):
            raise ValueError("There can't be any duplicated labels.")

        super().__init__(name=name, categories=categories, nan_freq=nan_freq, labels=labels)
        self.gender_model = GenderModel(name=f"{name}_gender", nan_freq=nan_freq, gender_label=labels.gender_label,
                                        title_label=labels.title_label)

        # Whether the gender can be learned from data or randomly sampled
        self.learn_gender = True if self.labels.gender_label or self.labels.title_label else False

        self.mobile_number_format = config.mobile_number_format
        self.home_number_format = config.home_number_format
        self.work_number_format = config.work_number_format
        self.dict_cache_size = config.dict_cache_size
        self.locale = config.locale
        self.provider = self.get_provider(self.locale)
        self.pwd_length = config.pwd_length

    def fit(self, df: pd.DataFrame) -> 'PersonModel':
        self.convert_df_for_children(df)
        if self.learn_gender:
            self.gender_model.fit(df)

        self._fitted = True
        self.revert_df_from_children(df)
        return self

    def sample(self, n: Optional[int] = None, produce_nans: bool = False,
               conditions: Optional[pd.DataFrame] = None) -> pd.DataFrame:

        if n is None and conditions is None:
            raise ValueError("One of 'n' or 'conditions' must be given.")

        if conditions is None:
            assert n is not None
            conditions = self.gender_model.sample(n, produce_nans=produce_nans)
        elif self.gender_model.name not in conditions.columns:
            conditions = self.gender_model.sample(len(conditions), produce_nans=produce_nans)
        return self._sample_conditional(conditions)

    def _sample_conditional(self, conditions: pd.DataFrame) -> pd.DataFrame:
        num_rows = len(conditions)
        if self.gender_model.name not in conditions:
            raise ValueError(f"Given dataframe doesn't contain column '{self.gender_model.name}' to "
                             "sample conditionally from.")

        gender = conditions.loc[:, self.gender_model.name]

        firstname = gender.astype(dtype=str).apply(self.generate_random_first_name)
        lastname = pd.Series(data=np.random.choice(Provider.last_names, size=num_rows))

        df = pd.DataFrame([[]] * num_rows)
        if self.labels.gender_label is not None:
            df.loc[:, self.labels.gender_label] = conditions.loc[:, self.labels.gender_label]
        if self.labels.title_label is not None:
            df.loc[:, self.labels.title_label] = conditions.loc[:, self.labels.title_label]
        if self.labels.name_label is not None:
            df.loc[:, self.labels.name_label] = firstname.str.cat(others=lastname, sep=' ')
        if self.labels.firstname_label is not None:
            df.loc[:, self.labels.firstname_label] = firstname
        if self.labels.lastname_label is not None:
            df.loc[:, self.labels.lastname_label] = lastname
        if self.labels.email_label is not None:
            # https://email-verify.my-addr.com/list-of-most-popular-email-domains.php
            # we don't want clashes with real emails
            # domain = np.random.choice(a=['gmail.com', 'yahoo.com', 'hotmail.com'], size=len(data))
            fkr = faker.Faker(locale=self.locale)
            df.loc[:, self.labels.email_label] = self.generate_usernames(firstname, lastname)
            df.loc[:, self.labels.email_label] = df.loc[:, self.labels.email_label].str.cat(
                others=[fkr.domain_name() for _ in range(num_rows)],
                sep='@')
            assert all(df.loc[:, self.labels.email_label].apply(self.check_email))

        if self.labels.username_label is not None:
            df.loc[:, self.labels.username_label] = self.generate_usernames(firstname, lastname)
        if self.labels.password_label is not None:
            df.loc[:, self.labels.password_label] = [self.generate_password(*self.pwd_length) for _ in range(num_rows)]
        if self.labels.mobile_number_label is not None:
            df.loc[:, self.labels.mobile_number_label] = [self.generate_phone_number(self.mobile_number_format)
                                                          for _ in range(num_rows)]
        if self.labels.home_number_label is not None:
            df.loc[:, self.labels.home_number_label] = [self.generate_phone_number(self.home_number_format)
                                                        for _ in range(num_rows)]
        if self.labels.work_number_label is not None:
            df.loc[:, self.labels.work_number_label] = [self.generate_phone_number(self.work_number_format)
                                                        for _ in range(num_rows)]

        columns = [c for c in self.labels.__dict__.values() if c is not None]
        df.loc[gender.isna(), columns] = np.nan
        return df

    @staticmethod
    def generate_usernames(firstname: pd.Series, lastname: pd.Series) -> pd.Series:
        username = firstname\
            .apply(lambda x: x + np.random.choice(['', '.', '-', '_']))\
            .str.cat(others=lastname)\
            .apply(lambda x: x + str(random.randint(0, 100) if random.random() < 0.5 else ''))

        while username.nunique() < len(firstname):
            vc = username.value_counts()
            duplicates = list(vc[vc > 1].index)

            username[username.isin(duplicates)] = username[username.isin(duplicates)].apply(
                lambda x: x + str(random.randint(0, 100)))

        return username.str.lower()

    @staticmethod
    def generate_password(pwd_length_min: int = 8, pwd_length_max: int = 16) -> str:
        pwd_length = random.randint(pwd_length_min, pwd_length_max)
        possible_chars = string.ascii_letters + string.digits
        return ''.join(random.choice(possible_chars) for _ in range(pwd_length))

    @staticmethod
    def generate_phone_number(number_format: str = '07xxxxxxxx') -> str:
        return re.sub(r'x', lambda _: str(random.randint(0, 9)), number_format)

    def generate_random_first_name(self, gender: str):
        if gender == 'M':
            return np.random.choice(self.provider.first_names_male)
        elif gender == 'F':
            return np.random.choice(self.provider.first_names_female)
        else:
            return np.random.choice(self.provider.first_names)

    @staticmethod
    def check_email(s: str) -> bool:
        m = re.match(r"""(?:[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b
        \x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z
        0-9])?\.)+[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}
        (?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[A-Za-z0-9-]*[A-Za-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f
        \x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])""", s)
        return True if m is not None else False

    @staticmethod
    def get_provider(locale):
        try:
            provider = importlib.import_module(f"faker.providers.person.{locale}")
        except ModuleNotFoundError:
            raise ValueError(f"Given locale '{locale}' not valid")
        return provider.Provider

    @classmethod
    def from_meta(cls, meta: Person, config: PersonModelConfig = PersonModelConfig()) -> 'PersonModel':
        return cls(name=meta.name, categories=meta.categories, nan_freq=meta.nan_freq, labels=meta.labels, config=config)
