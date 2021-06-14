import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from synthesized.insight.metrics.distance import EarthMoversDistanceBinned

logger = logging.getLogger(__name__)


class LinkageAttack:
    def __init__(
            self, df_orig: pd.DataFrame, t_closeness: float, k_distance: float, max_n_vulnerable: float,
            key_columns: Optional[List[str]] = None, sensitive_columns: Optional[List[str]] = None
    ):
        self._df_orig = df_orig
        self.t_closeness = t_closeness
        self.k_distance = k_distance
        self.max_n_vulnerable = max_n_vulnerable
        self._sensitive_columns: List[str] = []
        self._key_columns: List[str] = []
        self.key_columns = key_columns if key_columns is not None else []
        self.sensitive_columns = sensitive_columns if sensitive_columns is not None else []

    @property
    def df_orig(self) -> pd.DataFrame:
        return self._df_orig[self.columns].copy()

    @property
    def sensitive_columns(self) -> List[str]:
        return self._sensitive_columns.copy()

    @sensitive_columns.setter
    def sensitive_columns(self, cols: Union[None, List[str]]):
        if cols is None or len(cols) == 0:
            self._sensitive_columns = []
        else:
            if not pd.Index(cols).isin(self._df_orig.columns).all():
                raise ValueError("Sensitive columns must be in DataFrame.")

            if pd.Index(cols).isin(self.key_columns).any():
                raise ValueError("Sensitive columns can't also be key columns.")

            self._sensitive_columns = cols

    @property
    def key_columns(self) -> List[str]:
        return self._key_columns.copy()

    @key_columns.setter
    def key_columns(self, cols: Union[None, List[str]]):
        if cols is None or len(cols) == 0:
            self._key_columns = []
        else:
            if not pd.Index(cols).isin(self._df_orig.columns).all():
                raise ValueError("Key columns must be in DataFrame.")

            if pd.Index(cols).isin(self.sensitive_columns).any():
                raise ValueError("Key columns can't also be sensitive columns.")

            self._key_columns = cols

    @property
    def columns(self) -> List[str]:
        return self.key_columns + self._sensitive_columns

    def get_quantiles(self, df_attack: pd.DataFrame, n_bins: int) -> Dict[str, pd.Series]:
        quantiles = [round(i, 3) for i in np.arange(0, 1 + 1 / n_bins, 1 / n_bins)]
        col_quantiles = dict()
        for col in self.key_columns:
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

    def get_vulnerable(self, df: pd.DataFrame, limit_n: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        maxes = {col: np.nanmax(df[col]) for col in self.sensitive_columns}
        mins = {col: np.nanmin(df[col]) for col in self.sensitive_columns}

        t_closeness = []

        if limit_n:
            def incorrect_size(n):
                return n == 0 or n > self.max_n_vulnerable
        else:
            def incorrect_size(n):
                return n == 0

        def different_distribution(g) -> bool:
            diff = False
            for col in self.sensitive_columns:
                hist_df, bins = np.histogram(df[col], bins='auto', range=(mins[col], maxes[col]))
                hist_g, _ = np.histogram(g[col], bins=bins)
                dist = EarthMoversDistanceBinned(hist_g, hist_df, bins).distance
                norm = (maxes[col] - mins[col]) / 2

                if norm > 0 and dist / norm > self.t_closeness:
                    t_closeness.append(
                        {'sensitive_col': col, 't': dist / norm, **{col: g[col].iloc[0] for col in self.key_columns}})
                    diff = True

            return diff

        def is_vulnerable(g) -> bool:
            if incorrect_size(len(g)):
                return False

            return different_distribution(g)

        vuln = df.groupby(self.key_columns).filter(is_vulnerable).groupby(self.key_columns).aggregate(list)
        if len(vuln) == 0:
            t_df = pd.DataFrame()
        else:
            t_df = pd.DataFrame(t_closeness).pivot(index=self.key_columns, columns='sensitive_col', values='t')
        return vuln, t_df

    def get_attacks(self, df_attack: pd.DataFrame, n_bins: int = 25) -> Optional[pd.DataFrame]:
        quantiles = self.get_quantiles(df_attack[self.columns], n_bins)

        binned_df_orig = self.bin_df(df=self.df_orig, quantiles=quantiles)
        vulnerable_df_orig, t_orig = self.get_vulnerable(df=binned_df_orig, limit_n=True)
        binned_df_attack = self.bin_df(df=df_attack[self.columns].copy(), quantiles=quantiles)
        vulnerable_df_attack, t_attack = self.get_vulnerable(df=binned_df_attack, limit_n=False)

        maxes = {
            col: max(np.nanmax(binned_df_attack[col]), np.nanmax(binned_df_orig[col]))
            for col in self.sensitive_columns
        }
        mins = {
            col: np.nanmin([np.nanmin(binned_df_attack[col]), np.nanmin(binned_df_orig[col])])
            for col in self.sensitive_columns
        }

        attacked_data = []

        for row in vulnerable_df_attack.itertuples():
            key = row.Index
            if key not in vulnerable_df_orig.index:
                continue

            rows_attack = vulnerable_df_attack.loc[key]
            rows_orig = vulnerable_df_orig.loc[key]

            for col in self.sensitive_columns:
                if pd.isna([rows_attack[col]]).all() or pd.isna([rows_orig[col]]).all():
                    continue

                r_a = [r for r in rows_attack[col] if pd.notna(r)]
                r_o = [r for r in rows_orig[col] if pd.notna(r)]
                bins = np.histogram_bin_edges(r_a + r_o, bins='auto', range=(mins[col], maxes[col]))
                hist_a, _ = np.histogram(r_a, bins=bins)
                hist_o, _ = np.histogram(r_o, bins=bins)
                dist = EarthMoversDistanceBinned(pd.Series(hist_o), pd.Series(hist_a), bins).distance
                norm = (maxes[col] - mins[col]) / 2

                if dist / norm > self.k_distance or pd.isna(t_orig.loc[key, col]) or pd.isna(t_attack.loc[key, col]):
                    continue

                attacked_data.append({
                    'sensitive_column': col,
                    'original_values': rows_orig.loc[col],
                    'attack_values': rows_attack.loc[col],
                    'k_dist': dist / norm,
                    't_orig': t_orig.loc[key, col],
                    't_attack': t_attack.loc[key, col],
                    **{self.key_columns[n]: k for n, k in enumerate(key)}
                })

        if len(attacked_data) == 0:
            return None

        attacked_df = pd.DataFrame(
            attacked_data
        ).pivot(index=self.key_columns, columns='sensitive_column').swaplevel(0, 1, 1).sort_index(axis=1)

        return attacked_df
