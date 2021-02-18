import gzip
import importlib
import logging
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import faker
import numpy as np
import pandas as pd
import simplejson

from .histogram import Histogram
from ..base import Model
from ..base.value_meta import Nominal, NType
from ..value import Address
from ...config import AddressLabels, AddressModelConfig

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
    full_address: Optional[str] = None

    def __init__(self, postcode: Optional[str], county: Optional[str], city: Optional[str], district: Optional[str],
                 street: Optional[str], house_number: Optional[str], flat: Optional[str],
                 house_name: Optional[str]):
        self.postcode = postcode.replace("'", "") if postcode is not None else None
        self.county = county.replace("'", "") if county is not None else None
        self.city = city.replace("'", "") if city is not None else None
        self.district = district.replace("'", "") if district is not None else None
        self.street = street.replace("'", "") if street is not None else None
        self.house_number = house_number.replace("'", "") if house_number is not None else None
        self.flat = flat.replace("'", "") if flat is not None else None
        self.house_name = house_name.replace("'", "") if house_name is not None else None
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


class AddressModel(Address, Model):
    postcode_regex = re.compile(r'[A-Za-z]{1,2}[0-9]+[A-Za-z]? *[0-9]+[A-Za-z]{2}')

    def __init__(self, name, nan_freq: Optional[float] = None,
                 labels: AddressLabels = AddressLabels(),
                 config: AddressModelConfig = AddressModelConfig()):

        super().__init__(name=name, nan_freq=nan_freq, labels=labels)
        self.postcode_model = Histogram(name=f"{name}_postcode", nan_freq=self.nan_freq)

        if all([c is None for c in self.params.values()]):
            raise ValueError("At least one of labels must be given")
        if name in self.params.values():
            raise ValueError("Value of 'name' can't be equal to any other label.")

        self.locale = config.locale
        self.provider = self.get_provider(self.locale)
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
    def label_to_address_record_key(self) -> Dict[str, str]:
        d: Dict[str, str] = dict()
        for label, key in zip([self.labels.postcode_label, self.labels.full_address_label, self.labels.county_label,
                               self.labels.city_label, self.labels.district_label, self.labels.street_label,
                               self.labels.house_number_label, self.labels.flat_label, self.labels.house_name_label],
                              ['postcode', 'full_address', 'county',
                               'city', 'district', 'street',
                               'house_number', 'flat', 'house_name']):
            if label is not None:
                d[label] = key
        return d

    def fit(self, df: pd.DataFrame) -> 'AddressModel':

        if self.nan_freq is None:
            self.nan_freq = df[next(s for s in self.params.values() if s)].isna().sum()

        if self.fake:
            postcodes = self.provider.POSTAL_ZONES
            self.postcode_model.categories = postcodes
            self.postcode_model.probabilities = {c: 1 / len(postcodes) for c in postcodes}
            self._fitted = True
            return self

        categories = list(self.postcodes.keys())
        if self.learn_postcodes:
            postcode_sr = self._get_postcode_key_from_df(df)
            self.postcode_model.fit(postcode_sr.to_frame(self.postcode_model.name))

        else:
            self.postcode_model.categories = categories
            self.postcode_model.probabilities = {c: 1 / len(categories) for c in categories}
            self.postcode_model._fitted = True

        return self

    def sample(self, n: int, produce_nans: bool = False) -> pd.DataFrame:

        df = self.postcode_model.sample(n, produce_nans=produce_nans)
        df = self._sample_conditional(df, postcode_column=self.postcode_model.name)
        df.drop(columns=self.postcode_model.name, inplace=True)

        return df

    def sample_conditional(self, df: pd.DataFrame) -> pd.DataFrame:

        if self.labels.postcode_label is None:
            raise ValueError("This model can't sample conditionally as 'postcode_label' is not given.")
        if self.labels.postcode_label not in df.columns:
            raise ValueError(f"Can't sample conditionally from a dataframe without '{self.labels.postcode_label}'")

        return self._sample_conditional(df, postcode_column=self.labels.postcode_label)

    def _sample_conditional(self, df: pd.DataFrame, postcode_column: str) -> pd.DataFrame:

        if self.fake:
            n = len(df)
            address_records = self.generate_fake_address_records(n)
            for label, key in self.label_to_address_record_key.items():
                if label != postcode_column:
                    df[label] = [asdict(address_record)[key] for address_record in address_records]
            df.loc[df[postcode_column].isna(), self.params.values()] = np.nan

        else:
            def sample_address(postcode_key):
                if postcode_key in self.postcodes.keys():
                    return np.random.choice(self.postcodes[postcode_key])
                return None

            postcode_sr = df[postcode_column].fillna('nan').apply(self._get_postcode_key)
            address_records = np.vectorize(sample_address)(postcode_sr.dropna())

            columns = self.params.values()
            for c in columns:
                df[c] = np.nan

            for label, key in self.label_to_address_record_key.items():
                df.loc[df[postcode_column] != 'nan', label] = [
                    address_record.__dict__[key] if address_record else np.nan
                    for address_record in address_records
                ]
            isna = df.apply(lambda x: any([True if x[c] == 'nan' else False for c in columns]), axis=1)
            df.loc[isna, columns] = np.nan

        return df

    def _get_postcode_key_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.labels.postcode_label is not None:
            return df[self.labels.postcode_label].fillna('nan').apply(self._get_postcode_key)
        elif self.labels.full_address_label is not None:
            return df[self.labels.full_address_label].fillna('nan').apply(self._get_postcode_key_from_address)
        else:
            raise ValueError("Can't extract postcode if 'postcode_label' or 'full_address_label' are not defined.")

    def _get_postcode_key(self, postcode: str):
        if self.postcode_model.categories and postcode in self.postcode_model.categories:
            return postcode

        if postcode == 'nan':
            return 'nan'

        if not AddressModel.postcode_regex.match(postcode):
            return 'nan'
        if self.postcode_level == 0:  # 1-2 letters
            index = 2 - postcode[1].isdigit()
        elif self.postcode_level == 1:
            index = postcode.index(' ')
        elif self.postcode_level == 2:
            index = postcode.index(' ') + 2
        else:
            raise ValueError(self.postcode_level)
        return postcode[:index]

    def _load_postcodes_dict(self, addresses_file) -> Dict[str, np.ndarray]:  # Dict[str, List[AddressRecord]]

        d: Dict[str, List[AddressRecord]] = dict()

        if os.path.exists(addresses_file):
            with gzip.open(addresses_file, 'r') as f:
                for line in f:
                    js = simplejson.loads(line)

                    addresses = []
                    for js_i in js['addresses']:
                        addresses.append(AddressRecord(
                            postcode=js['postcode'],
                            county=js_i['county'],
                            city=js_i['town_or_city'],
                            district=js_i['district'],
                            street=js_i['thoroughfare'],
                            house_number=js_i['building_number'],
                            flat=js_i['building_name'] if js_i['building_name'] else js_i['sub_building_name'],
                            house_name=js_i['building_name']
                        ))
                    postcode_key = self._get_postcode_key(js['postcode'])
                    if postcode_key not in d.keys():
                        d[self._get_postcode_key(js['postcode'])] = addresses
                    else:
                        d[self._get_postcode_key(js['postcode'])].extend(addresses)

        # convert list to ndarray for better performance
        d_out: Dict[str, np.ndarray] = dict()
        for key, postcode in d.items():
            d_out[key] = np.array(postcode)

        return d_out

    def _get_postcode_key_from_address(self, x):
        g = self.postcode_regex.search(x)
        return self._get_postcode_key(g.group(0)) if g else 'nan'

    def check_null(self, value: str, label: Optional[str]) -> Optional[str]:
        return value if label or self.labels.full_address_label else None

    def generate_fake_address_records(self, n: int) -> List[AddressRecord]:
        fkr = faker.Faker(locale=self.locale)
        address_records = []
        for _ in range(n):
            postcode = self.check_null(fkr.postcode(), self.labels.postcode_label)
            county = self.check_null(fkr.county(), self.labels.county_label)
            city = self.check_null(fkr.city(), self.labels.city_label)
            district = self.check_null(fkr.city(), self.labels.district_label)
            street = self.check_null(fkr.street_name(), self.labels.street_label)
            house_number = self.check_null(fkr.building_number(), self.labels.house_number_label)
            flat = self.check_null(fkr.secondary_address(), self.labels.flat_label)
            house_name = self.check_null((fkr.last_name() + " " + fkr.street_suffix()), self.labels.house_name_label)

            address_records.append(
                AddressRecord(postcode=postcode, county=county, city=city, district=district, street=street,
                              house_number=house_number, flat=flat, house_name=house_name))

        return address_records

    @staticmethod
    def get_provider(locale):
        try:
            provider = importlib.import_module(f"faker.providers.address.{locale}")
        except ModuleNotFoundError:
            raise ValueError(f"Given locale '{locale}' not valid")
        return provider.Provider

    @classmethod
    def from_meta(cls, meta: Nominal[NType], config: AddressModelConfig = AddressModelConfig()) -> 'AddressModel':
        assert isinstance(meta, Address)
        return cls(name=meta.name, nan_freq=meta.nan_freq, labels=meta.labels, config=config)
