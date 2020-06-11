import re

import numpy as np
import pandas as pd

from .base_mask import BaseMask


class PartialMask(BaseMask):
    def __init__(self, column_name: str, masking_proportion: float = 0.75):
        super(PartialMask, self).__init__(column_name, assert_fitted=False)
        self.masking_proportion = masking_proportion

    def fit(self, df: pd.DataFrame):
        super(PartialMask, self).fit(df)

    def transform(self, df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
        df = super(PartialMask, self).transform(df, inplace)
        df.loc[:, self.column_name] = df.loc[:, self.column_name].astype(str).apply(self.mask_key)
        return df

    def mask_key(self, k):
        to_replace = int(np.ceil(len(k) * self.masking_proportion))
        regex = "^.{" + str(to_replace) + "}"
        return re.sub(regex, "x" * to_replace, k)
