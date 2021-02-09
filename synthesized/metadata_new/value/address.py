import logging
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .categorical import String
from ..base import Meta

logger = logging.getLogger(__name__)


class Address(String):
    """
    Address
    """

    def __init__(
            self, name, categories: Sequence[str] = None, nan_freq: float = None,
            num_rows: int = None, postcode_label: Optional[str] = None, county_label: Optional[str] = None,
            city_label: Optional[str] = None, district_label: Optional[str] = None,
            street_label: Optional[str] = None, house_number_label: Optional[str] = None,
            flat_label: Optional[str] = None, house_name_label: Optional[str] = None
    ):

        super().__init__(name=name, categories=categories, nan_freq=nan_freq, num_rows=num_rows)

        self.postcode_label = postcode_label
        self.county_label = county_label
        self.city_label = city_label
        self.district_label = district_label
        self.street_label = street_label
        self.house_number_label = house_number_label
        self.flat_label = flat_label
        self.house_name_label = house_name_label

        string_labels = [house_number_label, flat_label, house_name_label, street_label, district_label,
                         city_label, county_label, postcode_label]

        children: List[Meta] = [String(label) for label in string_labels if label is not None]

        self.children = children

    def extract(self, df: pd.DataFrame):
        super().extract(df=df)
        return self

    def expand(self, df: pd.DataFrame):
        if self.name not in df.columns:
            raise KeyError
        sr_collapsed_address = df[self.name]
        df[list(self.keys())] = sr_collapsed_address.astype(str).str.split("|", n=len(self.keys()) - 1, expand=True)

        df.drop(columns=self.name, inplace=True)

    def collapse(self, df: pd.DataFrame):
        df[self.name] = df[list(self.keys())[0]].str.cat(
            [df[k].astype('string') for k in list(self.keys())[1:]], sep="|", na_rep=''
        )
        df.drop(columns=list(self.keys()), inplace=True)

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            'postcode_label': self.postcode_label,
            'county_label': self.county_label,
            'city_label': self.city_label,
            'district_label': self.district_label,
            'street_label': self.street_label,
            'house_number_label': self.house_number_label,
            'flat_label': self.flat_label,
            'house_name_label': self.house_name_label,
        })
        return d
