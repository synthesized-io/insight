import re
import os
from typing import List, Dict, Optional, Union
import gzip
import logging

from dataclasses import dataclass
import faker
import numpy as np
import pandas as pd
import tensorflow as tf
import simplejson

from .categorical import CategoricalMeta
from .value_meta import ValueMeta

logger = logging.getLogger(__name__)


@dataclass
class AddressParams:
    postcode_label: Union[str, List[str], None] = None
    county_label: Union[str, List[str], None] = None
    city_label: Union[str, List[str], None] = None
    district_label: Union[str, List[str], None] = None
    street_label: Union[str, List[str], None] = None
    house_number_label: Union[str, List[str], None] = None
    flat_label: Union[str, List[str], None] = None
    house_name_label: Union[str, List[str], None] = None
    addresses_file: Optional[str] = None


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


class AddressMeta(ValueMeta):
    postcode_regex = re.compile(r'^[A-Za-z]{1,2}[0-9]+[A-Za-z]? *[0-9]+[A-Za-z]{2}$')

    def __init__(self, name, postcode_level: int = 0, postcode_label: str = None,
                 county_label: str = None, city_label: str = None, district_label: str = None, street_label: str = None,
                 house_number_label: str = None, flat_label: str = None, house_name_label: str = None,
                 addresses_file: str = None):

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

        # Check if given 'addresses_file' exist, otherwise set to None.
        if addresses_file is not None:
            if not os.path.exists(os.path.expanduser(addresses_file)):
                logger.warning("Given address file '{}' does not exist, using fake addresses".format(addresses_file))
                addresses_file = None
            else:
                addresses_file = os.path.expanduser(addresses_file)

        if addresses_file is None:
            self.fake = True
            self.fkr = faker.Faker(locale='en_GB')
            self.postcodes: Dict[str, List[AddressRecord]] = {}
            self.postcode = None
        else:
            self.fake = False
            logger.info("Loading address dictionary from '{}'".format(addresses_file))
            self.postcodes = self._load_postcodes_dict(addresses_file)

            assert postcode_label is not None
            self.postcode = CategoricalMeta(name=postcode_label)

        self.dtype = tf.int64
        assert self.fake or self.postcode

    def columns(self) -> List[str]:
        columns = [
            self.county_label, self.postcode_label, self.city_label, self.district_label,
            self.street_label, self.house_number_label, self.flat_label, self.house_name_label
        ]
        return np.unique([c for c in columns if c is not None]).tolist()

    def extract(self, df: pd.DataFrame) -> None:
        super().extract(df=df)

        if self.fake:
            return

        contains_nans = (df.loc[:, self.postcode_label].isna().sum() > 0)

        if len(self.postcodes) == 0:
            for n, row in df.dropna().iterrows():
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
            unique_postcodes = list(self.postcodes.keys())
            if contains_nans:
                unique_postcodes.append(np.nan)

            postcode_data = pd.DataFrame({self.postcode_label: unique_postcodes})
            self.postcode.extract(df=postcode_data)

    def learned_input_columns(self) -> List[str]:
        if self.fake and self.postcode is not None:
            return self.postcode.learned_input_columns()
        else:
            return []

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fake and self.postcode_label is not None:
            df.loc[:, self.postcode_label] = df.loc[:, self.postcode_label].fillna('nan')
            df.loc[:, self.postcode_label] = df.loc[:, self.postcode_label].apply(self._get_postcode_key)

            if self.postcode:
                df = self.postcode.preprocess(df=df)

        return super().preprocess(df=df)

    def _get_postcode_key(self, postcode: str):
        if postcode == 'nan':
            return 'nan'

        if not AddressMeta.postcode_regex.match(postcode):
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

    def _load_postcodes_dict(self, addresses_file) -> Dict[str, List[AddressRecord]]:

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

        return d

    def learned_output_columns(self) -> List[str]:
        if self.fake and self.postcode is not None:
            return self.postcode.learned_output_columns()
        else:
            return []

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
