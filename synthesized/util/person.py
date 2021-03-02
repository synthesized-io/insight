from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def get_gender_from_gender(gender: str, gender_mapping: Dict[str, List[str]]) -> str:
    gender = gender.strip().upper()
    for k, v in gender_mapping.items():
        if gender in map(str.upper, v):
            return k
    return np.nan


def get_gender_from_title(title: str, title_mapping: Dict[str, List[str]]) -> str:
    title = title.replace('.', '').strip().upper()
    for k, v in title_mapping.items():
        if title in map(str.upper, v):
            return k
    return np.nan


def get_title_from_gender(gender: str, gender_mapping: Dict[str, List[str]],
                          title_mapping: Dict[str, List[str]]) -> str:
    gender = get_gender_from_gender(gender, gender_mapping)
    return title_mapping[gender][0] if gender in title_mapping.keys() else np.nan


def get_gender_from_df(df: pd.DataFrame, name: str, gender_label: Optional[str],
                       title_label: Optional[str], gender_mapping: Dict[str, List[str]],
                       title_mapping: Dict[str, List[str]]) -> pd.DataFrame:
    if gender_label is not None:
        df[name] = df[gender_label].astype(str).apply(get_gender_from_gender, gender_mapping=gender_mapping)
    elif title_label is not None:
        df[name] = df[title_label].astype(str).apply(get_gender_from_title, title_mapping=title_mapping)
    else:
        raise ValueError("Can't extract gender series as 'gender_label' nor 'title_label' are given.")

    return df


def get_gender_title_from_df(df: pd.DataFrame, name: str, gender_label: Optional[str],
                             title_label: Optional[str], gender_mapping: Dict[str, List[str]],
                             title_mapping: Dict[str, List[str]]) -> pd.DataFrame:
    if gender_label is not None:
        df[gender_label] = df[name]
    if title_label is not None:
        df[title_label] = df[name].astype(dtype=str).apply(get_title_from_gender, gender_mapping=gender_mapping,
                                                           title_mapping=title_mapping)

    return df
