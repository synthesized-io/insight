from typing import Dict

import pandas as pd

from .base_mask import BaseMask
from .null import NullMask
from .partial import PartialMask
from .random import RandomMask
from .rounding import RoundingMask
from .swapping import SwappingMask


class DataMasker:
    def __init__(self, masked_columns: Dict[str, str]):

        self.masked_columns = masked_columns
        self.transformers: Dict[str, BaseMask] = dict()
        for masked_column, masking_technique in self.masked_columns.items():

            if masking_technique.startswith('random'):
                split = masking_technique.split('|', 1)
                str_length = 10 if len(split) == 1 else int(split[1])
                self.transformers[masked_column] = RandomMask(column_name=masked_column, str_length=str_length)

            elif masking_technique.startswith('partial_masking'):
                split = masking_technique.split('|', 1)
                masking_proportion = .75 if len(split) == 1 else float(split[1])
                self.transformers[masked_column] = PartialMask(column_name=masked_column,
                                                               masking_proportion=masking_proportion)

            elif masking_technique.startswith('rounding'):
                split = masking_technique.split('|', 1)
                n_bins = 20 if len(split) == 1 else int(split[1])
                self.transformers[masked_column] = RoundingMask(column_name=masked_column, n_bins=n_bins)

            elif masking_technique == 'swapping':
                self.transformers[masked_column] = SwappingMask(column_name=masked_column)

            elif masking_technique == 'null':
                self.transformers[masked_column] = NullMask(column_name=masked_column)

            else:
                raise ValueError(f"Given masking technique '{masking_technique}' for "
                                 f"column '{masked_column}' not supported")

    def fit(self, df: pd.DataFrame):

        for masked_column, masker in self.transformers.items():
            masker.fit(df)

    def transform(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        if not inplace:
            df = df.copy()

        for masked_column, masker in self.transformers.items():
            df = masker.transform(df, inplace=True)

        return df

    def fit_transform(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df, inplace=inplace)
