from typing import List, Union

from dataclasses import dataclass
import faker
import numpy as np
import pandas as pd

from .categorical import CategoricalMeta
from .value_meta import ValueMeta


@dataclass
class PersonParams:
    title_label: Union[str, List[str], None] = None
    gender_label: Union[str, List[str], None] = None
    name_label: Union[str, List[str], None] = None
    firstname_label: Union[str, List[str], None] = None
    lastname_label: Union[str, List[str], None] = None
    email_label: Union[str, List[str], None] = None
    mobile_number_label: Union[str, List[str], None] = None
    home_number_label: Union[str, List[str], None] = None
    work_number_label: Union[str, List[str], None] = None


class PersonMeta(ValueMeta):

    def __init__(self, name, title_label=None, gender_label=None, name_label=None,
                 firstname_label=None, lastname_label=None, email_label=None,
                 mobile_number_label=None, home_number_label=None, work_number_label=None,
                 dict_cache_size=10000):
        super().__init__(name=name)

        self.title_label = title_label
        self.gender_label = gender_label
        self.name_label = name_label
        self.firstname_label = firstname_label
        self.lastname_label = lastname_label
        self.email_label = email_label
        self.mobile_number_label = mobile_number_label
        self.home_number_label = home_number_label
        self.work_number_label = work_number_label
        # Assume the gender are always encoded like M or F or U(???)
        self.title_mapping = {'M': 'MR', 'F': 'MRS', 'U': 'MRS'}
        self.gender_mapping = {'MR': 'M', 'MRS': 'F', 'MS': 'F', 'MISS': 'F'}

        fkr = faker.Faker(locale='en_GB')
        self.fkr = fkr
        self.male_first_name_cache = np.array(list({fkr.first_name_male() for _ in range(dict_cache_size)}))
        self.female_first_name_cache = np.array(list({fkr.first_name_female() for _ in range(dict_cache_size)}))
        self.last_name_cache = np.array(list({fkr.last_name() for _ in range(dict_cache_size)}))

        if gender_label is None:
            self.gender = None
        elif self.title_label == self.gender_label:
            # a special case when we extract gender from title:
            self.gender = CategoricalMeta(name=self.title_label + '_gender')
        else:
            self.gender = CategoricalMeta(name=gender_label)

    def columns(self) -> List[str]:
        columns = [
            self.title_label, self.gender_label, self.name_label, self.firstname_label, self.lastname_label,
            self.email_label, self.mobile_number_label, self.home_number_label, self.work_number_label
        ]
        return np.unique([c for c in columns if c is not None]).tolist()

    def learned_input_columns(self) -> List[str]:
        if self.gender is None:
            return []
        else:
            return self.gender.learned_input_columns()

    def learned_output_columns(self) -> List[str]:
        if self.gender is None:
            return []
        else:
            return self.gender.learned_output_columns()

    def extract(self, df: pd.DataFrame) -> None:
        super().extract(df=df)

        if self.gender is not None:
            if self.title_label == self.gender_label:
                df[self.title_label + '_gender'] = df[self.title_label].map(self.gender_mapping)
            self.gender.extract(df=df)
            if self.title_label == self.gender_label:
                df.drop(self.title_label + '_gender', axis=1, inplace=True)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.gender is not None:
            if self.title_label == self.gender_label:
                df[self.title_label + '_gender'] = df[self.title_label].map(self.gender_mapping)
            df = self.gender.preprocess(df=df)
        return super().preprocess(df=df)

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().postprocess(df=df)

        if self.gender is None:
            gender = pd.Series(np.random.choice(a=['F', 'M'], size=len(df)))
        else:
            df = self.gender.postprocess(df=df)
            if self.title_label == self.gender_label:
                gender = df[self.title_label + '_gender']
                df.drop(self.title_label + '_gender', axis=1, inplace=True)
            else:
                gender = df[self.gender_label]

        def get_first_name(g):
            if g == 'M':
                return np.random.choice(self.male_first_name_cache)
            else:
                return np.random.choice(self.female_first_name_cache)

        title = gender.astype(dtype=str).map(self.title_mapping)
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
        if self.mobile_number_label is not None:
            df.loc[:, self.mobile_number_label] = [self.fkr.cellphone_number() for _ in range(len(df))]
        if self.home_number_label is not None:
            df.loc[:, self.home_number_label] = [self.fkr.cellphone_number() for _ in range(len(df))]
        if self.work_number_label is not None:
            df.loc[:, self.work_number_label] = [self.fkr.cellphone_number() for _ in range(len(df))]
        return df
