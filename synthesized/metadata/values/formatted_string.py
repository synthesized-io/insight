from typing import List

import pandas as pd
import rstr

from .value_meta import ValueMeta
from ...config import FormattedStringMetaConfig


class FormattedStringMeta(ValueMeta):

    def __init__(self, name: str, formatted_string_label: str = None, regex: str = None,
                 config: FormattedStringMetaConfig = FormattedStringMetaConfig()):
        super().__init__(name=name)

        check_regex(regex)
        self.formatted_string_label: str = formatted_string_label if formatted_string_label is not None else name
        assert self.formatted_string_label is not None

        if regex is None:
            # If regex is not given, check if we can get regex from config.label_to_regex
            if config.label_to_regex is None or self.formatted_string_label not in config.label_to_regex:
                raise ValueError("All 'formatted_string_labels' must have a regex defined in config.label_to_regex.")
            self.regex: str = config.label_to_regex[self.formatted_string_label]

        else:
            self.regex = regex

    def extract(self, df: pd.DataFrame) -> None:
        pass

    def learned_input_columns(self) -> List[str]:
        return []

    def learned_output_columns(self) -> List[str]:
        return []

    def columns(self) -> List[str]:
        assert self.formatted_string_label is not None
        return [self.formatted_string_label]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        for c in self.columns():
            if c in df.columns:
                df.drop(c, axis=1, inplace=True)

        return super().preprocess(df)

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().postprocess(df=df)

        df[self.formatted_string_label] = [generate_str_from_regex(self.regex) for _ in range(len(df))]

        return df


def generate_str_from_regex(regex):
    return rstr.xeger(regex)


def check_regex(regex):
    try:
        generate_str_from_regex(regex)
    except Exception as e:
        raise ValueError(f"Given regex '{regex}' not valid: {e.msg}.")
