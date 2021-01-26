import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from pyemd import emd_samples

logger = logging.getLogger(__name__)


class LinkageAttack:
    def __init__(self, df_orig: pd.DataFrame, t_closeness: float, k_distance: float, max_n_vulnerable: float):
        self._df_orig = df_orig
        self.t_closeness = t_closeness
        self.k_distance = k_distance
        self.max_n_vulnerable = max_n_vulnerable

        self._key_columns: List[str] = []
        self._sensitive_columns: List[str] = []

    @property
    def df_orig(self) -> pd.DataFrame:
        return self._df_orig[self.columns].copy()

    @property
    def sensitive_columns(self) -> List[str]:
        return self._sensitive_columns

    @property
    def key_columns(self) -> List[str]:
        return self._key_columns

    @property
    def columns(self) -> List[str]:
        return self.key_columns + self._sensitive_columns

    @staticmethod
    def get_quantiles(df_attack: pd.DataFrame, n_bins: int) -> Dict[str, pd.Series]:
        quantiles = [round(i, 3) for i in np.arange(0, 1 + 1 / n_bins, 1 / n_bins)]
        col_quantiles = dict()
        for col in df_attack.columns:
            if df_attack[col].dtypes.kind in ('f', 'i') and df_attack[col].nunique() > n_bins:
                col_quantiles[col] = df_attack[col].quantile(quantiles).unique()

        return col_quantiles

    @staticmethod
    def bin_df(df: pd.DataFrame, quantiles: Dict[str, pd.Series]) -> pd.DataFrame:

        for col, quant in quantiles.items():
            binned = pd.cut(df[col], bins=quant, include_lowest=True)
            binned = binned.map({a: (a.right + a.left) / 2 for a in binned.cat.categories})

            df[col] = binned

        return df

    def get_vulnerable(self, df: pd.DataFrame, limit_n: bool = True) -> pd.DataFrame:
        maxes = {col: np.nanmax(df[col]) for col in self.sensitive_columns}
        mins = {col: np.nanmin(df[col]) for col in self.sensitive_columns}

        if limit_n:
            def incorrect_size(n):
                return n == 0 or n > self.max_n_vulnerable
        else:
            def incorrect_size(n):
                return n == 0

        def different_distribution(g) -> bool:
            for col in self.sensitive_columns:
                dist = emd_samples(g[col], df[col], range=(mins[col], maxes[col]))
                norm = (maxes[col] - mins[col])

                if norm > 0 and dist / norm > self.t_closeness:
                    return True

            return False

        def is_vulnerable(g) -> bool:
            if incorrect_size(len(g)):
                return False

            return different_distribution(g)

        return df.groupby(self.key_columns).filter(is_vulnerable).groupby(self.key_columns).aggregate(list)

    def get_attacks(self, df_attack: pd.DataFrame, n_bins: int = 25) :
        quantiles = self.get_quantiles(df_attack[self.columns], n_bins)

        binned_df_orig = self.bin_df(df=self.df_orig, quantiles=quantiles)
        vulnerable_df_orig = self.get_vulnerable(df=binned_df_orig, limit_n=True)

        binned_df_attack = self.bin_df(df=df_attack[self.columns].copy(), quantiles=quantiles)
        vulnerable_df_attack = self.get_vulnerable(df=binned_df_attack, limit_n=False)

        maxes = {col: np.nanmax(binned_df_attack[col]) for col in self.sensitive_columns}
        mins = {col: np.nanmin(binned_df_attack[col]) for col in self.sensitive_columns}

        attacked_data = []

        for row in vulnerable_df_attack.itertuples():
            key = row.Index
            if key in vulnerable_df_orig.index:
                rows_attack = vulnerable_df_attack.loc[key]
                rows_orig = vulnerable_df_orig.loc[key]

                for col in self.sensitive_columns:
                    dist = emd_samples(rows_attack[col], rows_orig[col], range=(mins[col], maxes[col]))
                    norm = (maxes[col] - mins[col])
                    if dist < self.k_distance:
                        attacked_data.append((rows_orig.loc[[col, ]], dist))

        return attacked_data
