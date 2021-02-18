import logging
from typing import Dict, Optional, Sequence

import pandas as pd

from .categorical import String
from ...config import PersonParams

logger = logging.getLogger(__name__)


class Person(String):
    """Person meta."""
    dtype = 'U'

    def __init__(
            self, name: str, categories: Optional[Sequence[str]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None, title_label: Optional[str] = None, gender_label: Optional[str] = None,
            full_name_label: Optional[str] = None, first_name_label: Optional[str] = None,
            last_name_label: Optional[str] = None, email_label: Optional[str] = None,
            username_label: Optional[str] = None, password_label: Optional[str] = None,
            mobile_number_label: Optional[str] = None, home_number_label: Optional[str] = None,
            work_number_label: Optional[str] = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq, num_rows=num_rows)

        self.title_label = title_label
        self.gender_label = gender_label
        self.full_name_label = full_name_label
        self.first_name_label = first_name_label
        self.last_name_label = last_name_label
        self.email_label = email_label
        self.username_label = username_label
        self.password_label = password_label
        self.mobile_number_label = mobile_number_label
        self.home_number_label = home_number_label
        self.work_number_label = work_number_label

        self.children = [
            String(child_label)
            for child_label in [
                title_label, gender_label, full_name_label, first_name_label, last_name_label, email_label,
                username_label, password_label, mobile_number_label, home_number_label, work_number_label
            ] if child_label is not None
        ]

    @classmethod
    def from_params(cls, params: PersonParams) -> 'Person':
        ann = Person(
            name=params.name, title_label=params.title_label, gender_label=params.gender_label,
            full_name_label=params.fullname_label, first_name_label=params.firstname_label,
            last_name_label=params.lastname_label, email_label=params.email_label,
            username_label=params.username_label, password_label=params.password_label,
            mobile_number_label=params.mobile_number_label, home_number_label=params.home_number_label,
            work_number_label=params.work_number_label
        )
        return ann

    def extract(self, df: pd.DataFrame):
        super().extract(df)

        return self

    def convert_df_for_children(self, df: pd.DataFrame):
        if self.name not in df.columns:
            raise KeyError
        sr_collapsed_address = df[self.name]
        df[list(self.keys())] = sr_collapsed_address.astype(str).str.split("|", n=len(self.keys()) - 1, expand=True)

        df.drop(columns=self.name, inplace=True)

    def revert_df_from_children(self, df: pd.DataFrame):
        df[self.name] = df[list(self.keys())[0]].astype(str).str.cat(
            [df[k].astype(str) for k in list(self.keys())[1:]], sep="|", na_rep=''
        )
        df.drop(columns=list(self.keys()), inplace=True)

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "title_label": self.title_label,
            "gender_label": self.gender_label,
            "full_name_label": self.full_name_label,
            "first_name_label": self.first_name_label,
            "last_name_label": self.last_name_label,
            "email_label": self.email_label,
            "username_label": self.username_label,
            "password_label": self.password_label,
            "mobile_number_label": self.mobile_number_label,
            "work_number_label": self.work_number_label,
            "home_number_label": self.home_number_label
        })

        return d
