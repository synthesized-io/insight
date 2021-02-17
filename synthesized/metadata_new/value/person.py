import logging
from typing import Dict, List, Sequence

import pandas as pd

from .categorical import String
from ..base import Meta
from ...config import PersonLabels

logger = logging.getLogger(__name__)


class Person(String):
    """
    Person
    """

    def __init__(
            self, name, categories: Sequence[str] = None, nan_freq: float = None,
            num_rows: int = None, labels: PersonLabels = PersonLabels()
    ):

        super().__init__(name=name, categories=categories, nan_freq=nan_freq, num_rows=num_rows)

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

        string_labels = [self.title_label, self.gender_label, self.name_label, self.firstname_label,
                         self.lastname_label, self.email_label, self.username_label, self.password_label,
                         self.mobile_number_label, self.home_number_label, self.work_number_label]

        children: List[Meta] = [String(label) for label in string_labels if label is not None]

        self.children = children

    def extract(self, df: pd.DataFrame):
        super().extract(df=df)
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
            'title_label': self.title_label,
            'gender_label': self.gender_label,
            'name_label': self.name_label,
            'firstname_label': self.firstname_label,
            'lastname_label': self.lastname_label,
            'email_label': self.email_label,
            'username_label': self.username_label,
            'password_label': self.password_label,
            'mobile_number_label': self.mobile_number_label,
            'home_number_label': self.home_number_label,
            'work_number_label': self.work_number_label,
        })
        return d
