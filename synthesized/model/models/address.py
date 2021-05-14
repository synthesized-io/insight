import gzip
import importlib
import logging
import os
import re
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Pattern, Sequence, cast

import faker
import numpy as np
import pandas as pd
import simplejson

from .histogram import Histogram
from ..base import DiscreteModel
from ...config import AddressLabels, AddressModelConfig, AddressRecord, PostcodeModelConfig
from ...metadata import MetaNotExtractedError
from ...metadata.value import Address, String
from ...util import get_postcode_key, get_postcode_key_from_df

logger = logging.getLogger(__name__)


class PostcodeModel(Histogram[str]):
    def __init__(self, meta: String, probabilities: Optional[Dict[str, float]] = None,
                 postcode_label: Optional[str] = None, full_address_label: Optional[str] = None,
                 config: PostcodeModelConfig = PostcodeModelConfig()):

        super().__init__(meta=meta, probabilities=probabilities)
        self.postcode_label = postcode_label
        self.full_address_label = full_address_label
        self.postcode_regex: Pattern[str] = re.compile(config.postcode_regex)

    def fit(self, df: pd.DataFrame) -> 'PostcodeModel':
        try:
            categories: Optional[Sequence[str]] = self.categories
        except MetaNotExtractedError:
            categories = None
        postcode_sr = get_postcode_key_from_df(
            df, postcode_regex=self.postcode_regex,
            postcode_label=self.postcode_label, full_address_label=self.full_address_label,
            postcodes=categories)
        super().fit(postcode_sr.to_frame(self.name))

        return self

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "postcode_label": self.postcode_label,
            "full_address_label": self.full_address_label,
            "postcode_regex": self.postcode_regex.pattern
        })

        return d

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> 'PostcodeModel':
        meta_dict = cast(Dict[str, object], d["meta"])
        meta = String.from_dict(meta_dict)
        model = cls(
            meta=meta, probabilities=cast(Optional[Dict[str, float]], d["probabilities"]),
            postcode_label=cast(Optional[str], d["postcode_label"]),
            full_address_label=cast(Optional[str], d["full_address_label"]),
            config=PostcodeModelConfig(cast(str, d["postcode_regex"]))
        )
        model._fitted = cast(bool, d["fitted"])

        return model


class AddressModel(DiscreteModel[Address, str]):
    def __init__(self, meta: Address, config: AddressModelConfig = AddressModelConfig()):
        super().__init__(meta=meta)
        postcode_meta = String(name=f"{meta.name}_postcode", nan_freq=meta.nan_freq, num_rows=meta.num_rows)
        self.config = config
        self.postcode_model = PostcodeModel(
            meta=postcode_meta, postcode_label=self.labels.postcode_label,
            full_address_label=self.labels.full_address_label, config=config.postcode_model_config
        )

        self.locale = config.address_locale
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

        else:
            self.fake = False
            logger.info("Loading address dictionary from '{}'".format(addresses_file))
            self.postcodes = self._load_postcodes_dict(addresses_file)
            postcodes = list(self.postcodes.keys())

        self.postcode_model.meta.categories = postcodes
        self.postcode_model.probabilities = {c: 1 / len(postcodes) for c in postcodes}

        if self.nan_freq:
            self._meta._extracted = self.postcode_model._meta._extracted = True
            self._fitted = self.postcode_model._fitted = True

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

    @property
    def labels(self) -> AddressLabels:
        return self._meta.labels

    @property
    def params(self) -> Dict[str, str]:
        return self._meta.params

    @property
    def categories(self) -> Sequence[str]:
        return self._meta.categories

    def fit(self, df: pd.DataFrame) -> 'AddressModel':
        super().fit(df=df)
        if self.nan_freq is None:
            self._meta.nan_freq = df[next(s for s in self.params.values() if s)].isna().sum()

        if self.fake:
            postcodes = self.provider.POSTAL_ZONES
            self.postcode_model.meta.categories = postcodes
            self.postcode_model.probabilities = {c: 1 / len(postcodes) for c in postcodes}

            return self

        if self.learn_postcodes:
            with self._meta.unfold(df):
                self.postcode_model.fit(df)

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
                lambda x: x.item() if isinstance(x, np.str_) else x
            ).apply(
                get_postcode_key,
                postcode_regex=self.postcode_model.postcode_regex, postcode_level=self.postcode_level,
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
                    try:
                        postcodes: Optional[Sequence[str]] = self.postcode_model.categories
                    except MetaNotExtractedError:
                        postcodes = None

                    postcode_key = get_postcode_key(
                        js['postcode'], postcode_regex=self.postcode_model.postcode_regex,
                        postcode_level=self.postcode_level, postcodes=postcodes
                    )

                    if postcode_key not in d.keys():
                        d[get_postcode_key(
                            js['postcode'], postcode_regex=self.postcode_model.postcode_regex,
                            postcode_level=self.postcode_level, postcodes=postcodes
                        )] = addresses
                    else:
                        d[get_postcode_key(
                            js['postcode'], postcode_regex=self.postcode_model.postcode_regex,
                            postcode_level=self.postcode_level, postcodes=postcodes
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

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "config": asdict(self.config)
        })

        return d

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> 'AddressModel':
        meta_dict = cast(Dict[str, object], d["meta"])
        meta = Address.from_dict(meta_dict)
        config = cast(Dict[str, Any], d["config"])
        model = cls(meta=meta, config=AddressModelConfig(**config))
        model._fitted = cast(bool, d["fitted"])

        return model


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
