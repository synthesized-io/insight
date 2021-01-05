import gzip
import logging
import os
import re
from typing import Dict, List

import faker
import numpy as np
import pandas as pd
import simplejson
import tensorflow as tf

from ...config import AddressMetaConfig
from .categorical import CategoricalMeta
from .value_meta import ValueMeta

logger = logging.getLogger(__name__)


class AddressRecord:
    def __init__(self, postcode, county, city, district, street, house_number, flat, house_name):
        self.postcode = postcode.replace("'", "") if postcode is not None else None
        self.county = county.replace("'", "") if county is not None else None
        self.city = city.replace("'", "") if city is not None else None
        self.district = district.replace("'", "") if district is not None else None
        self.street = street.replace("'", "") if street is not None else None
        self.house_number = house_number.replace("'", "") if house_number is not None else None
        self.flat = flat.replace("'", "") if flat is not None else None
        self.house_name = house_name.replace("'", "") if house_name is not None else None

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

    @property
    def full_address(self):
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


class AddressMeta(ValueMeta):
    postcode_regex = re.compile(r'[A-Za-z]{1,2}[0-9]+[A-Za-z]? *[0-9]+[A-Za-z]{2}')

    def __init__(self, name, postcode_level: int = 0, postcode_label: str = None,
                 county_label: str = None, city_label: str = None, district_label: str = None, street_label: str = None,
                 house_number_label: str = None, flat_label: str = None, house_name_label: str = None,
                 full_address_label: str = None,
                 config: AddressMetaConfig = AddressMetaConfig()):

        super().__init__(name=name)

        if postcode_level < 0 or postcode_level > 2:
            raise NotImplementedError

        self.postcode_level = postcode_level
        self.postcode_label = postcode_label
        self.county_label = county_label
        self.city_label = city_label
        self.district_label = district_label
        self.street_label = street_label
        self.house_number_label = house_number_label
        self.flat_label = flat_label
        self.house_name_label = house_name_label
        self.full_address_label = full_address_label

        self.config = config

        addresses_file = config.addresses_file
        # Check if given 'addresses_file' exist, otherwise set to None.
        if addresses_file is not None:
            if not os.path.exists(os.path.expanduser(addresses_file)):
                logger.warning("Given address file '{}' does not exist, using fake addresses".format(addresses_file))
                addresses_file = None
            else:
                addresses_file = os.path.expanduser(addresses_file)

        # Generate fake addresses
        if addresses_file is None:
            self.fake: bool = True
            self.postcodes: Dict[str, List[AddressRecord]] = {}
            self.postcode = None

        # Generate real addresses from random postcode
        elif (self.postcode_label is None and self.full_address_label is None) or not self.config.learn_postcodes:
            self.fake = False
            logger.info("Loading address dictionary from '{}'".format(addresses_file))
            self.postcodes = self._load_postcodes_dict(addresses_file)
            self.postcode = None

        # Generate real addresses and learn postcode
        else:
            self.fake = False
            logger.info("Loading address dictionary from '{}'".format(addresses_file))
            self.postcodes = self._load_postcodes_dict(addresses_file)
            name = postcode_label or full_address_label
            assert name is not None
            self.postcode = CategoricalMeta(name=name)

        self.dtype = tf.int64

    def columns(self) -> List[str]:
        columns = [
            self.county_label, self.postcode_label, self.city_label, self.district_label,
            self.street_label, self.house_number_label, self.flat_label, self.house_name_label,
            self.full_address_label
        ]
        return np.unique([c for c in columns if c is not None]).tolist()

    def extract(self, df: pd.DataFrame) -> None:
        super().extract(df=df)

        if self.fake:
            return

        if self.postcode_label or self.full_address_label:
            contains_nans = (df.loc[:, self.postcode_label or self.full_address_label].isna().sum() > 0)

        if len(self.postcodes) == 0:
            for n, row in df.dropna().iterrows():
                if self.postcode_label is not None:
                    postcode = row[self.postcode_label]

                # If we don't have postcode but we have full address, extract postcode via regex
                elif self.full_address_label is not None:
                    postcode = row[self.full_address_label]
                    m = re.search(self.postcode_regex, postcode)
                    if not m:
                        contains_nans = True
                        continue
                    postcode = m.group(0)

                else:
                    raise ValueError("One of 'postcode_label' and 'full_address_label' must be given")

                postcode_key = self._get_postcode_key(postcode)
                if postcode_key == 'nan':
                    contains_nans = True
                    continue

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

            postcode_data = pd.DataFrame({self.postcode_label or self.full_address_label: unique_postcodes})
            self.postcode.extract(df=postcode_data)

    def learned_input_columns(self) -> List[str]:
        if self.postcode is not None:
            return self.postcode.learned_input_columns()
        else:
            return []

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:

        if self.postcode_label is not None:
            df.loc[:, self.postcode_label] = df.loc[:, self.postcode_label].fillna('nan')
            df.loc[:, self.postcode_label] = df.loc[:, self.postcode_label].apply(self._get_postcode_key)

        elif self.full_address_label is not None:
            df.loc[:, self.full_address_label] = df.loc[:, self.full_address_label].fillna('nan')
            df.loc[:, self.full_address_label] = df.loc[:, self.full_address_label].apply(self.get_postcode)
            df.loc[:, self.full_address_label] = \
                df.loc[:, self.full_address_label].apply(self._get_postcode_key)

        if self.postcode:
            df = self.postcode.preprocess(df=df)

        return super().preprocess(df=df)

    def _get_postcode_key(self, postcode: str):
        if postcode == 'nan':
            return 'nan'

        if not AddressMeta.postcode_regex.match(postcode):
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
        if self.postcode is not None:
            return self.postcode.learned_output_columns()
        else:
            return []

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().postprocess(df=df)
        if self.fake:
            address_records = self.generate_fake_address_records(len(df))

            if self.full_address_label:
                df[self.full_address_label] = [address_record.full_address for address_record in address_records]

            if self.postcode_label is not None:
                df[self.postcode_label] = [address_record.postcode for address_record in address_records]

            if self.county_label is not None:
                df[self.county_label] = [address_record.county for address_record in address_records]

            if self.city_label is not None:
                df[self.city_label] = [address_record.city for address_record in address_records]

            if self.district_label is not None:
                df[self.district_label] = [address_record.district for address_record in address_records]

            if self.street_label is not None:
                df[self.street_label] = [address_record.street for address_record in address_records]

            if self.house_number_label is not None:
                df[self.house_number_label] = [address_record.house_number for address_record in address_records]

            if self.flat_label is not None:
                df[self.flat_label] = [address_record.flat for address_record in address_records]

            if self.house_name_label is not None:
                df[self.house_name_label] = [address_record.house_name for address_record in address_records]

        else:
            if self.postcode is None:
                postcode = np.random.choice(a=list(self.postcodes), size=len(df))
            else:
                df = self.postcode.postprocess(df=df)
                postcode = df[self.postcode_label or self.full_address_label].astype(dtype='str').to_numpy()

            def sample_address(postcode_key):
                if postcode_key in self.postcodes.keys():
                    return np.random.choice(self.postcodes[postcode_key])
                return np.nan

            addresses = np.vectorize(sample_address)(postcode)

            if self.postcode_label is not None:
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

            if self.full_address_label is not None:
                df[self.full_address_label] = list(map(lambda a: a.full_address, addresses))

        return df

    def get_postcode(self, x):
        g = self.postcode_regex.search(x)
        return g.group(0) if g else 'nan'

    def generate_fake_address_records(self, n: int) -> List[AddressRecord]:
        fkr = faker.Faker(locale='en_GB')
        address_records = []
        for _ in range(n):
            postcode = fkr.postcode() if self.postcode_label or self.full_address_label else None
            county = fkr.county() if self.county_label or self.full_address_label else None
            city = fkr.city() if self.city_label or self.full_address_label else None
            district = fkr.city() if self.district_label or self.full_address_label else None
            street = fkr.street_name() if self.street_label or self.full_address_label else None
            house_number = fkr.building_number() \
                if (self.house_number_label or self.full_address_label) and np.random.random() < 0.3 else None
            flat = fkr.secondary_address() \
                if (self.flat_label or self.full_address_label) and np.random.random() < 0.3 else None
            house_name = (fkr.last_name() + " " + fkr.street_suffix()) \
                if (self.house_name_label or self.full_address_label) and np.random.random() < 0.3 else None

            address_records.append(
                AddressRecord(postcode=postcode, county=county, city=city, district=district, street=street,
                              house_number=house_number, flat=flat, house_name=house_name))

        return address_records
