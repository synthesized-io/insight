import gzip
import importlib
import logging
import os
import re
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Pattern, Sequence

import faker
import numpy as np
import pandas as pd
import simplejson

from .histogram import Histogram
from ..base import Model
from ...config import AddressLabels, AddressModelConfig, PostcodeModelConfig
from ...metadata_new import Nominal, NType
from ...metadata_new.value import Address
from ...util import get_postcode_key, get_postcode_key_from_df

logger = logging.getLogger(__name__)


@dataclass(repr=True)
class AddressRecord:
    postcode: Optional[str] = None
    county: Optional[str] = None
    city: Optional[str] = None
    district: Optional[str] = None
    street: Optional[str] = None
    house_number: Optional[str] = None
    flat: Optional[str] = None
    house_name: Optional[str] = None
    full_address: str = field(init=False, repr=False)

    def __post_init__(self):
        for f in fields(self):
            if f.name != 'full_address':
                field_val = getattr(self, f.name)
                setattr(self, f.name, field_val.replace("'", "") if field_val is not None else None)
        self.full_address = self.compute_full_address()

    def compute_full_address(self) -> str:
        address_str = ""
        if self.flat:
            address_str += f"{self.flat} "
        if self.house_number:
            address_str += f"{self.house_number} "
        if self.house_name:
            address_str += f"{self.house_name}, "
        if self.street:
            address_str += f"{self.street}, "
        if self.district:
            address_str += f"{self.district}, "
        if self.postcode:
            address_str += f"{self.postcode} "
        if self.city:
            address_str += f"{self.city} "
        if self.county:
            address_str += f"{self.county}"

        return address_str


class PostcodeModel(Histogram[str]):
    def __init__(self, name, categories: Optional[Sequence[str]] = None, nan_freq: Optional[float] = None,
                 postcode_label: Optional[str] = None, full_address_label: Optional[str] = None,
                 config: PostcodeModelConfig = PostcodeModelConfig()):

        super().__init__(name=name, categories=categories, nan_freq=nan_freq)
        self.postcode_label = postcode_label
        self.full_address_label = full_address_label
        self.postcode_regex = config.postcode_regex

    def fit(self, df: pd.DataFrame) -> 'PostcodeModel':
        postcode_sr = get_postcode_key_from_df(
            df, postcode_regex=self.postcode_regex,
            postcode_label=self.postcode_label, full_address_label=self.full_address_label,
            postcodes=self.categories)
        super().fit(postcode_sr.to_frame(self.name))
        return self


class AddressModel(Address, Model):
    postcode_regex: Pattern[str] = re.compile(r'[A-Za-z]{1,2}[0-9]+[A-Za-z]? *[0-9]+[A-Za-z]{2}')

    def __init__(self, name: str, categories: Optional[Sequence[str]] = None, nan_freq: Optional[float] = None,
                 labels: AddressLabels = AddressLabels(), config: AddressModelConfig = AddressModelConfig()):

        super().__init__(name=name, categories=categories, nan_freq=nan_freq, labels=labels)
        self.postcode_model = PostcodeModel(
            name=f"{name}_postcode", nan_freq=self.nan_freq, postcode_label=labels.postcode_label,
            full_address_label=labels.full_address_label, config=config.postcode_model_config)

        if all([c is None for c in self.params.values()]):
            raise ValueError("At least one of labels must be given")
        if name in self.params.values():
            raise ValueError("Value of 'name' can't be equal to any other label.")

        self.locale = config.locale
        self.provider = self._get_provider(self.locale)
        self.postcode_level = config.postcode_level
        if self.postcode_level < 0 or self.postcode_level > 2:
            raise NotImplementedError
        self.learn_postcodes = config.learn_postcodes
        if self.labels.postcode_label is None and self.labels.full_address_label is None:
            self.learn_postcodes = False

        addresses_file = config.addresses_file
        # Check if given 'addresses_file' exist, otherwise set to None.
        if addresses_file is not None:
            if not os.path.exists(os.path.expanduser(addresses_file)):
                logger.warning(f"Given address file '{addresses_file}' does not exist, using fake addresses")
                addresses_file = None
            else:
                addresses_file = os.path.expanduser(addresses_file)

        if addresses_file is None:
            self.fake: bool = True
            self.postcodes: Dict[str, List[AddressRecord]] = {}
            # Manually 'fit' the postcode model so that it is not needed to call .fit().
            postcodes = self.provider.POSTAL_ZONES
            self.postcode_model.categories = postcodes
            self.postcode_model.probabilities = {c: 1 / len(postcodes) for c in postcodes}
            if self.nan_freq:
                self._extracted = self.postcode_model._extracted = True
                self._fitted = self.postcode_model._fitted = True

        else:
            self.fake = False
            logger.info("Loading address dictionary from '{}'".format(addresses_file))
            self.postcodes = self._load_postcodes_dict(addresses_file)

    @property
    def address_record_key_to_label(self) -> Dict[str, str]:
        d: Dict[str, str] = dict()
        for key, label in zip(['postcode', 'full_address', 'county',
                               'city', 'district', 'street',
                               'house_number', 'flat', 'house_name'],
                              [self.labels.postcode_label, self.labels.full_address_label, self.labels.county_label,
                               self.labels.city_label, self.labels.district_label, self.labels.street_label,
                               self.labels.house_number_label, self.labels.flat_label, self.labels.house_name_label]):
            if label is not None:
                d[key] = label
        return d

    def fit(self, df: pd.DataFrame) -> 'AddressModel':

        if self.nan_freq is None:
            self.nan_freq = df[next(s for s in self.params.values() if s)].isna().sum()

        if self.fake:
            postcodes = self.provider.POSTAL_ZONES
            self.postcode_model.categories = postcodes
            self.postcode_model.probabilities = {c: 1 / len(postcodes) for c in postcodes}
            self._fitted = self.postcode_model._fitted = True
            return self

        if self.learn_postcodes:
            self.postcode_model.fit(df)

        else:
            categories = list(self.postcodes.keys())
            self.postcode_model.categories = categories
            self.postcode_model.probabilities = {c: 1 / len(categories) for c in categories}
            self.postcode_model._fitted = True

        self._fitted = True
        return self

    def sample(self, n: Optional[int], produce_nans: bool = False,
               conditions: Optional[pd.DataFrame] = None) -> pd.DataFrame:

        if n is None and conditions is None:
            raise ValueError("One of 'n' or 'conditions' must be given.")

        if conditions is None:
            assert n is not None
            conditions = self.postcode_model.sample(n, produce_nans=produce_nans)

        elif self.postcode_model.name not in conditions.columns:
            n = len(conditions)
            conditions[self.postcode_model.name] = self.postcode_model.sample(n, produce_nans=produce_nans)

        df = self._sample_conditional(conditions)
        return df

    def _sample_conditional(self, conditions: pd.DataFrame) -> pd.DataFrame:
        assert self.postcode_model.name in conditions.columns

        columns = self.params.values()

        if self.fake:
            n = len(conditions)
            address_records = self._generate_fake_address_records(n)
            df = pd.DataFrame(address_records).rename(columns=self.address_record_key_to_label)[columns]

            # Add NaNs
            df.loc[conditions[self.postcode_model.name].isna(), self.params.values()] = np.nan

        else:
            def sample_address(postcode_key):
                if postcode_key in self.postcodes.keys():
                    return np.random.choice(self.postcodes[postcode_key])
                return AddressRecord()

            postcode_sr = conditions[self.postcode_model.name].fillna('nan').apply(
                get_postcode_key,
                postcode_regex=self.postcode_regex, postcode_level=self.postcode_level,
                postcodes=self.postcode_model.categories)
            address_records = list(np.vectorize(sample_address)(postcode_sr))

            df = pd.DataFrame(address_records).rename(columns=self.address_record_key_to_label)[columns]
            df.loc[conditions[self.postcode_model.name].isna(), columns] = np.nan

        return df

    def _load_postcodes_dict(self, addresses_file) -> Dict[str, np.ndarray]:  # Dict[str, List[AddressRecord]]

        d: Dict[str, List[AddressRecord]] = dict()

        if os.path.exists(addresses_file):
            with gzip.open(addresses_file, 'r') as f:
                for line in f:
                    js = simplejson.loads(line)
                    addresses = _create_address_records_with_labels(js, self.labels)

                    postcode_key = get_postcode_key(
                        js['postcode'], postcode_regex=self.postcode_regex, postcode_level=self.postcode_level,
                        postcodes=self.postcode_model.categories)
                    if postcode_key not in d.keys():
                        d[get_postcode_key(
                            js['postcode'], postcode_regex=self.postcode_regex,
                            postcode_level=self.postcode_level, postcodes=self.postcode_model.categories
                        )] = addresses
                    else:
                        d[get_postcode_key(
                            js['postcode'], postcode_regex=self.postcode_regex,
                            postcode_level=self.postcode_level, postcodes=self.postcode_model.categories
                        )].extend(addresses)

        # convert list to ndarray for better performance
        d_out: Dict[str, np.ndarray] = dict()
        for key, postcode in d.items():
            d_out[key] = np.array(postcode)

        return d_out

    def _check_null(self, generate_value: Callable[[], str], label: Optional[str]) -> Optional[str]:
        return generate_value() if label or self.labels.full_address_label else None

    def _generate_fake_address_records(self, n: int) -> List[AddressRecord]:
        fkr = faker.Faker(locale=self.locale)
        address_records = []
        for _ in range(n):
            postcode = self._check_null(fkr.postcode, self.labels.postcode_label)
            county = self._check_null(fkr.county, self.labels.county_label)
            city = self._check_null(fkr.city, self.labels.city_label)
            district = self._check_null(fkr.city, self.labels.district_label)
            street = self._check_null(fkr.street_name, self.labels.street_label)
            house_number = self._check_null(fkr.building_number, self.labels.house_number_label)
            flat = self._check_null(fkr.secondary_address, self.labels.flat_label)
            house_name = self._check_null(lambda: f"{fkr.last_name()} {fkr.street_suffix()}",
                                          self.labels.house_name_label)

            address_records.append(
                AddressRecord(postcode=postcode, county=county, city=city, district=district, street=street,
                              house_number=house_number, flat=flat, house_name=house_name))

        return address_records

    @staticmethod
    def _get_provider(locale):
        try:
            provider = importlib.import_module(f"faker.providers.address.{locale}")
        except ModuleNotFoundError:
            raise ValueError(f"Given locale '{locale}' not valid")
        return provider.Provider

    @classmethod
    def from_meta(cls, meta: Nominal[NType], config: AddressModelConfig = AddressModelConfig()) -> 'AddressModel':
        assert isinstance(meta, Address)
        return cls(name=meta.name, categories=meta.categories, nan_freq=meta.nan_freq,
                   labels=meta.labels, config=config)


def _create_address_records(js: Dict[str, Any]) -> List[AddressRecord]:
    addresses = []
    for js_i in js['addresses']:
        addresses.append(AddressRecord(
            postcode=js['postcode'], county=js_i['county'], city=js_i['town_or_city'],
            district=js_i['district'], street=js_i['thoroughfare'], house_number=js_i['building_number'],
            flat=js_i['building_name'] if js_i['building_name'] else js_i['sub_building_name'],
            house_name=js_i['building_name']
        ))
    return addresses


def _create_address_records_with_labels(js: Dict[str, Any], labels: AddressLabels) -> List[AddressRecord]:
    if labels.full_address_label is not None:
        return _create_address_records(js=js)

    addresses = []
    postcode = js['postcode'] if labels.postcode_label else None

    for js_i in js['addresses']:
        addresses.append(_create_record(postcode=postcode, js_i=js_i, labels=labels))

    return addresses


def _create_record(postcode: str, js_i: Dict[str, Any], labels: AddressLabels) -> AddressRecord:
    return AddressRecord(
        postcode=postcode,
        county=js_i['county'] if labels.county_label else None,
        city=js_i['town_or_city'] if labels.city_label else None,
        district=js_i['district'] if labels.district_label else None,
        street=js_i['thoroughfare'] if labels.street_label else None,
        house_number=js_i['building_number'] if labels.house_number_label else None,
        flat=(js_i['building_name'] or js_i['sub_building_name']) if labels.flat_label else None,
        house_name=js_i['building_name'] if labels.house_name_label else None,
    )
