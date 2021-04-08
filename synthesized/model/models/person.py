import importlib
import random
import re
import string
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, cast

import faker
import numpy as np
import pandas as pd
from faker.providers.person.en import Provider

from .histogram import Histogram
from ..base import DiscreteModel
from ..exceptions import ModelNotFittedError
from ...config import PersonLabels, PersonModelConfig
from ...metadata import MetaNotExtractedError
from ...metadata.value import Person, String
from ...util.person import collections_from_mapping


class GenderModel(Histogram[str]):
    """A Model of the gender attribute.

    This models gender with 4(+1) possible categories:
        * "F"  - female
        * "M"  - male
        * "NB" - non-binary
        * "A"  - ambiguous
        * (<NA> - missing value)

    The 4 correspond to collections of values in `self.meta.categories` and are calculated using 3 regex patterns,
    1 each for "F", "M", and "NB".

    Values that are detected by more than one regex are placed in the collection for ambiguous values, "A". This can
    happen when, gender is modelled off another attribute, for example the title attribute. i.e.
        * ["Mrs", "Ms"]  -> "F"
        * ["Mr"]         -> "M"
        * ["Ind", "Mx"]  -> "NB"
        * ["Dr", "Prof"] -> "A"

    """
    key_female = "F"
    key_male = "M"
    key_non_binary = "NB"
    key_ambiguous = "A"

    def __init__(
            self, meta: String, probabilities: Dict[str, float] = None, regex_female: str = r"^(mrs|ms).?$",
            regex_male: str = r'^(mr).?$', regex_non_binary: str = r"^(ind|mx|per).?$"
    ):
        super().__init__(meta=meta, probabilities=probabilities)
        self.regex_female = regex_female
        self.regex_male = regex_male
        self.regex_non_binary = regex_non_binary

    @property
    def collections(self) -> Dict[str, List[str]]:
        mapping = {
            self.key_female: self.regex_female,
            self.key_male: self.regex_male,
            self.key_non_binary: self.regex_non_binary
        }
        collections = collections_from_mapping(self.meta.categories, mapping, self.key_ambiguous)
        return {key: collection for key, collection in collections.items() if len(collection) > 0}

    @property
    def categories(self) -> Sequence[str]:
        return list(self.collections.keys())

    def fit(self, df):
        try:
            value_map = {v: key for key, collection in self.collections.items() for v in collection}
        except MetaNotExtractedError:
            self.meta.extract(df)
            value_map = {v: key for key, collection in self.collections.items() for v in collection}

        df_gender = df[[self.name]].copy()
        df_gender[self.name] = df_gender[self.name].astype(str).map(value_map)

        return super().fit(df_gender)

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "regex_female": self.regex_female,
            "regex_male": self.regex_male,
            "regex_non_binary": self.regex_non_binary
        })

        return d

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> 'GenderModel':
        meta_dict = cast(Dict[str, object], d["meta"])
        meta = String.from_dict(meta_dict)
        model = cls(
            meta=meta,
            probabilities=cast(Optional[Dict[str, float]], d["probabilities"]),
        )
        model._fitted = cast(bool, d["fitted"])

        return model


class PersonModel(DiscreteModel[Person, str]):
    """A Model of Person which captures gender with a hidden model.

    This model can be used to create the following attributes of Person:
        * `gender` (orig.)
        * `title` (orig.)
        * `first_name`
        * `last_name`
        * `email`
        * `username`
        * `password`
        * `home/work/mobile_number`

    Attributes marked with 'orig.' have values that correspond to the original dataset. The rest are intelligently
    generated based on the hidden model for the hidden attribute, '_gender' = {"F", "M", "NB", "A"}.

    There are 3 special configuration cases for this model that should be considered:

        1. The attribute `gender` is present: In this case the hidden model for `_gender` is based directly on the
            `gender` attribute. All values in the `gender` attribute should correspond to "F", "M", "U" or <NA>.
            In other words, there should be no ambiguous values in the collection "A".

        2. No `gender` present but `title` is present: The hidden model for `_gender` can be based on the available
            titles. As this is not a direct correspondance, not all values will correspond a single collection. In
            other words, there MAY be some ambiguous values in the collection "A".

        3. Neither `gender` nor `title` is present: The hidden model for gender cannot be fitted to the data and so
            the '_gender' attribute is assumed to be evenly distributed amongst the genders specified in the config.


    """
    def __init__(self, meta: Person, config: PersonModelConfig = PersonModelConfig()):
        super().__init__(meta=meta)
        self.config = config

        if self.labels.gender_label:
            self.hidden_model: Histogram = GenderModel(
                meta=meta[self.labels.gender_label], regex_female=self.config.gender_female_regex,
                regex_male=self.config.gender_male_regex, regex_non_binary=self.config.gender_non_binary_regex
            )
        elif self.labels.title_label:
            self.hidden_model = GenderModel(
                meta=meta[self.labels.title_label], regex_female=self.config.title_female_regex,
                regex_male=self.config.title_male_regex, regex_non_binary=self.config.title_non_binary_regex
            )
        else:
            gender_meta = String(f"{self.name}_gender", categories=self.config.genders, nan_freq=0)
            self.hidden_model = Histogram(
                meta=gender_meta,
                probabilities={gender: 1 / len(self.config.genders) for gender in self.config.genders}
            )

        self.mobile_number_format = config.mobile_number_format
        self.home_number_format = config.home_number_format
        self.work_number_format = config.work_number_format
        self.dict_cache_size = config.dict_cache_size
        self.locale = config.locale
        self.provider = self.get_provider(self.locale)
        self.pwd_length = config.pwd_length

    @property
    def categories(self) -> Sequence[str]:
        if self._meta.categories is None:
            raise ModelNotFittedError

        return self._meta.categories

    @property
    def labels(self) -> PersonLabels:
        return self._meta.labels

    @property
    def params(self) -> Dict[str, str]:
        return self._meta.params

    def fit(self, df: pd.DataFrame) -> 'PersonModel':
        super().fit(df=df)
        with self._meta.unfold(df=df):
            if isinstance(self.hidden_model, GenderModel):
                self.hidden_model.fit(df)

        return self

    def sample(self, n: Optional[int] = None, produce_nans: bool = False,
               conditions: Optional[pd.DataFrame] = None) -> pd.DataFrame:

        if n is None and conditions is None:
            raise ValueError("One of 'n' or 'conditions' must be given.")

        if conditions is None:
            assert n is not None
            conditions = self.hidden_model.sample(n, produce_nans=produce_nans)
        elif self.hidden_model.name not in conditions.columns:
            conditions = self.hidden_model.sample(len(conditions), produce_nans=produce_nans)

        return self._sample_conditional(conditions)

    def _sample_conditional(self, conditions: pd.DataFrame) -> pd.DataFrame:
        num_rows = len(conditions)
        if self.hidden_model.name not in conditions:
            raise ValueError(f"Given dataframe doesn't contain column '{self.hidden_model.name}' to "
                             "sample conditionally from.")

        gender = conditions.loc[:, self.hidden_model.name].astype(dtype=str)
        df = pd.DataFrame([[]] * num_rows)

        if self.labels.gender_label is not None:
            label = self.labels.gender_label
            mapping = {
                GenderModel.key_female: self.config.gender_female_regex,
                GenderModel.key_male: self.config.gender_male_regex,
                GenderModel.key_non_binary: self.config.gender_non_binary_regex
            }
            # There should be no ambiguous values if the hidden model is based on gender.
            collections = collections_from_mapping(self.meta[label].categories, mapping, None)
            df.loc[:, label] = gender.astype(dtype=str).apply(self.generate_from_collections, collections=collections)

        if self.labels.title_label is not None:
            label = self.labels.title_label
            mapping = {
                GenderModel.key_female: self.config.title_female_regex,
                GenderModel.key_male: self.config.title_male_regex,
                GenderModel.key_non_binary: self.config.title_non_binary_regex
            }
            collections = collections_from_mapping(self.meta[label].categories, mapping, GenderModel.key_ambiguous)
            df.loc[:, label] = gender.apply(self.generate_from_collections, collections=collections)
            # If hidden model isn't directly based on gender, "A" could exist and must be randomly resampled.
            categories = list(mapping.keys())
            gender = gender.apply(lambda x: random.choice(categories) if x == GenderModel.key_ambiguous else x)

        firstname = gender.apply(self.generate_random_first_name)
        lastname = pd.Series(data=np.random.choice(Provider.last_names, size=num_rows))

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
            .apply(lambda x: x + np.random.choice(['', '.', '-', '_']) if pd.notna(x) else '')\
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
        if gender == GenderModel.key_male:
            return np.random.choice(self.provider.first_names_male)
        elif gender == GenderModel.key_female:
            return np.random.choice(self.provider.first_names_female)
        elif gender == GenderModel.key_non_binary:
            return np.random.choice(self.provider.first_names)
        else:
            return np.nan

    @staticmethod
    def generate_from_collections(key: str, collections: Dict[str, List[str]]):
        categories = collections.get(key, [])
        if len(categories) == 0:
            return np.nan
        return np.random.choice(categories)

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

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "config": asdict(self.config)
        })

        return d

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> 'PersonModel':
        meta_dict = cast(Dict[str, object], d["meta"])
        meta = Person.from_dict(meta_dict)
        config = cast(Dict[str, Any], d["config"])
        model = cls(meta=meta, config=PersonModelConfig(**config))
        model._fitted = cast(bool, d["fitted"])

        return model
