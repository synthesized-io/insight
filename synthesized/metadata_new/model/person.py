import random
import re
import string
from typing import Any, List, Optional

import faker
import numpy as np
import pandas as pd
from faker.providers.person.en import Provider

from .histogram import Histogram
from ..base.value_meta import Nominal, NType
from ..value import Person
from ...config import PersonLabels, PersonModelConfig


class PersonModel(Histogram):

    def __init__(self, name, nan_freq: Optional[float] = None,
                 labels: PersonLabels = PersonLabels(),
                 config: PersonModelConfig = PersonModelConfig()):

        self.title_label = labels.title_label
        self.gender_label = labels.gender_label
        self.name_label = labels.name_label
        self.firstname_label = labels.firstname_label
        self.lastname_label = labels.lastname_label
        self.email_label = labels.email_label
        self.username_label = labels.username_label
        self.password_label = labels.password_label
        self.mobile_number_label = labels.mobile_number_label
        self.home_number_label = labels.home_number_label
        self.work_number_label = labels.work_number_label

        if all([c is None for c in self.columns]):
            raise ValueError("At least one of labels must be given")
        if name in self.columns:
            raise ValueError("Value of 'name' can't be equal to any other label.")

        # Assume the gender are always encoded like M or F or U(???)
        self.title_mapping = {'M': 'MR', 'F': 'MS', 'U': 'MX'}
        self.gender_mapping = {'MR': 'M', 'MRS': 'F', 'MS': 'F', 'MISS': 'F', 'MX': 'U'}
        self.titles = list(self.title_mapping.values())
        self.genders = list(self.title_mapping.keys())

        super().__init__(name=name, categories=self.genders,
                         probabilities={gender: 1 / len(self.genders) for gender in self.genders},
                         nan_freq=nan_freq)

        self.mobile_number_format = config.mobile_number_format
        self.home_number_format = config.home_number_format
        self.work_number_format = config.work_number_format

        self.dict_cache_size = config.dict_cache_size

    @property
    def columns(self) -> List[str]:
        columns = [
            self.title_label, self.gender_label, self.name_label, self.firstname_label, self.lastname_label,
            self.email_label, self.username_label, self.password_label,
            self.mobile_number_label, self.home_number_label, self.work_number_label
        ]
        return list(np.unique([c for c in columns if c is not None]))

    def fit(self, df: pd.DataFrame) -> 'PersonModel':
        if self.gender_label or self.title_label:
            gender_sr = self.get_gender_series(df)
            super().fit(gender_sr.to_frame(self.name))

        return self

    def sample(self, n: int, produce_nans: bool = False) -> pd.DataFrame:
        df = super().sample(n, produce_nans=produce_nans)
        return self.sample_conditional(df)

    def sample_conditional(self, df: pd.DataFrame) -> pd.DataFrame:
        fkr = faker.Faker(locale='en_GB')
        num_rows = len(df)

        gender = df.loc[:, self.name].astype(dtype=str)
        title = gender.astype(dtype=str).map(self.title_mapping)

        firstname = gender.astype(dtype=str).apply(self.generate_random_first_name)
        lastname = pd.Series(data=np.random.choice(Provider.last_names, size=num_rows))

        if self.gender_label is not None:
            df.loc[:, self.gender_label] = gender
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
            df.loc[:, self.email_label] = self.generate_usernames(firstname, lastname)
            df.loc[:, self.email_label] = df.loc[:, self.email_label].str.cat(
                others=[fkr.domain_name() for _ in range(num_rows)],
                sep='@')
            assert all(df.loc[:, self.email_label].apply(self.check_email))

        if self.username_label is not None:
            df.loc[:, self.username_label] = self.generate_usernames(firstname, lastname)
        if self.password_label is not None:
            df.loc[:, self.password_label] = [self.generate_password() for _ in range(num_rows)]
        if self.mobile_number_label is not None:
            df.loc[:, self.mobile_number_label] = [self.generate_phone_number(self.mobile_number_format)
                                                   for _ in range(num_rows)]
        if self.home_number_label is not None:
            df.loc[:, self.home_number_label] = [self.generate_phone_number(self.home_number_format)
                                                 for _ in range(num_rows)]
        if self.work_number_label is not None:
            df.loc[:, self.work_number_label] = [self.generate_phone_number(self.work_number_format)
                                                 for _ in range(num_rows)]

        df.loc[df[self.name].isna(), self.columns] = np.nan
        df.drop(columns=self.name, inplace=True)
        return df

    def probability(self, x: Any) -> float:
        raise NotImplementedError

    def get_gender_series(self, df: pd.DataFrame) -> pd.Series:
        if self.gender_label is not None:
            return df[self.gender_label].astype(str).str.upper()
        elif self.title_label is not None:
            return df[self.title_label].astype(str).str.upper().map(self.gender_mapping)
        else:
            raise ValueError("Can't extract gender series as 'gender_label' is not given.")

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

    @staticmethod
    def generate_random_first_name(gender: str):
        if gender == 'M':
            return np.random.choice(Provider.first_names_male)
        elif gender == 'F':
            return np.random.choice(Provider.first_names_female)
        else:
            return np.random.choice(Provider.first_names)

    @staticmethod
    def check_email(s: str) -> bool:
        m = re.match(r"""(?:[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b
        \x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z
        0-9])?\.)+[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}
        (?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[A-Za-z0-9-]*[A-Za-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f
        \x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])""", s)
        return True if m is not None else False

    @classmethod
    def from_meta(cls, meta: Nominal[NType], config: PersonModelConfig = PersonModelConfig()) -> 'PersonModel':
        assert isinstance(meta, Person)
        labels = PersonLabels(
            title_label=meta.title_label,
            gender_label=meta.gender_label,
            name_label=meta.name_label,
            firstname_label=meta.firstname_label,
            lastname_label=meta.lastname_label,
            email_label=meta.email_label,
            username_label=meta.username_label,
            password_label=meta.password_label,
            mobile_number_label=meta.mobile_number_label,
            home_number_label=meta.home_number_label,
            work_number_label=meta.work_number_label,
        )
        return cls(name=meta.name, nan_freq=meta.nan_freq, labels=labels, config=config)
