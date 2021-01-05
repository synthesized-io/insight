from typing import List

import pandas as pd
import rstr

from ...config import FormattedStringMetaConfig
from .value_meta import ValueMeta


class FormattedStringMeta(ValueMeta):

    def __init__(self, name: str, formatted_string_label: str = None,
                 regex: str = None, length: str = None, characters: str = None,
                 config: FormattedStringMetaConfig = FormattedStringMetaConfig()):
        super().__init__(name=name)

        self.formatted_string_label: str = formatted_string_label if formatted_string_label is not None else name
        assert self.formatted_string_label is not None

        if regex is not None:
            self.regex = regex

        elif length is not None and characters is not None:
            self.regex = r'[%s]{%s}' % (characters, length)

        # If regex is not given, check if we can get regex from config.label_to_regex
        elif config.label_to_regex is not None and self.formatted_string_label in config.label_to_regex:
            self.regex = config.label_to_regex[self.formatted_string_label]

        else:
            raise ValueError(f"Could not get regex for formatted string {name}.")

        check_regex(self.regex)

    def __str__(self):
        return "formatted_string"

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
        raise ValueError(f"Given regex '{regex}' not valid: {e}.")
