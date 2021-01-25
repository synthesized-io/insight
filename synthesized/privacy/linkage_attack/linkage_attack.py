from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ...insight.metrics import earth_movers_distance
from ...metadata import MetaExtractor


class LinkageAttack:
    """Given some prior knowledge about key columns, a LinkageAttack can be performed if some information can be
    extracted from the sensitive columns of the original dataset.
    """
    def __init__(self, df_original: pd.DataFrame, key_columns: List[str], sensitive_columns: List[str]):
        """Linkage attack constructor
        Args:
            df_original: Original dataset.
            key_columns: Key columns used to link information between original and attacker datasets.
            sensitive_columns: Columns that contain sensitive information. The attack is performed to obtain this
                information.
        """

        self.df_original = df_original.copy()
        self.key_columns = key_columns
        self.sensitive_columns = sensitive_columns
        self.n_bins = 25
        self.n_samples_vulnerable = 25
        self.t_closeness = 0.3
        self.k_distance = 0.05

        self.columns = df_original.columns
        self.n_columns = len(self.columns)

        self.df_original_bin = self.get_df_bin(df_original)
        self.df_meta = MetaExtractor.extract(self.df_original_bin)
        self.vulnerable_rows = self.get_vulnerable_rows(self.df_original_bin)
        self.vulnerable_rows_indexes = list(self.vulnerable_rows.groupby(self.key_columns).count().index)

        self.key_columns_idx = [i for i in range(self.n_columns) if self.columns[i] in self.key_columns]

    def get_attacks(self, df_attacker: pd.DataFrame) -> List[Dict]:
        """Given a dataset with previous knowledge, attack the original dataset and try to extract sensitive
        information linking key columns and checking how similar the sensitive information is in linked subsets.
        Args:
             df_attacker: Attacker DataFrame containing the previous knowledge about the original data.
        Returns:
            A tuple of dictionaries containing extracted information.
        """

        df_attacker_bin = self.get_df_bin(df_attacker)
        vulnerable_rows_synth = self.get_vulnerable_rows(df_attacker_bin)

        attacks = []
        for row in vulnerable_rows_synth.itertuples(index=False):
            key_columns_row = [row[i] for i in self.key_columns_idx]

            if tuple(key_columns_row) in self.vulnerable_rows_indexes:

                rows_orig = self.get_rows(self.df_original_bin, key_columns_row)
                rows_attacker = self.get_rows(df_attacker_bin, key_columns_row)

                sensitive_columns_attacked = []
                for sensitive_column in self.sensitive_columns:

                    dist = earth_movers_distance(rows_orig[sensitive_column], rows_attacker[sensitive_column], dp=self.df_meta)
                    if dist is not None and dist < self.k_distance:
                        sensitive_columns_attacked.append((sensitive_column, dist))

                if len(sensitive_columns_attacked) > 0:
                    attack = self.get_attack_dict(key_columns_row, sensitive_columns_attacked,
                                                  rows_known=rows_attacker.index,
                                                  rows_attacked=rows_orig.index)
                    attacks.append(attack)

        return attacks

    def get_rows(self, df: pd.DataFrame, key_columns_value: List[Any]) -> pd.DataFrame:
        return df.loc[
            np.prod([df[key_column] == key_columns_value[i] for i, key_column in enumerate(self.key_columns)],
                    axis=0).astype(bool)
        ]

    def get_df_bin(self, df: pd.DataFrame) -> pd.DataFrame:
        df_bin = df.copy()

        quantiles = [round(i, 3) for i in np.arange(0, 1 + 1 / self.n_bins, 1 / self.n_bins)]

        for c in self.columns:
            column = df_bin[c]
            if column.dtypes.kind in ('f', 'i') and column.nunique() > self.n_bins:
                quantiles_c = df_bin[c].quantile(quantiles)
                df_bin[c] = pd.cut(df_bin[c], bins=quantiles_c, include_lowest=True, duplicates='drop')

                df_bin[c] = df_bin[c].astype(str)

        return df_bin

    def get_vulnerable_rows(self, df: pd.DataFrame) -> pd.DataFrame:

        def is_vulnerable(g):
            len_g = len(g)
            if len_g == 0 or len_g > self.n_samples_vulnerable:
                return False

            for sensitive_column in self.sensitive_columns:
                dist = earth_movers_distance(g[sensitive_column], df[sensitive_column])
                if dist > self.t_closeness:
                    return True

            return False

        return df.groupby(self.key_columns).filter(is_vulnerable)

    def get_attack_dict(self, key_columns_row: List[Any], sensitive_columns_attacked: List[Tuple[str, float]],
                        rows_known: List[int], rows_attacked: List[int]) -> Dict[str, Any]:
        return dict(
            key_columns_values={self.key_columns[i]: key_columns_row[i] for i in range(len(self.key_columns))},
            attacked_attr_distances={attacked_attr: distance for attacked_attr, distance in sensitive_columns_attacked},
            rows_known=rows_known,
            rows_attacked=rows_attacked
        )

    def get_attacked_rows(self, attacks: List[Dict[str, Any]]) -> pd.DataFrame:
        if len(attacks) == 0:
            return pd.DataFrame([], columns=self.columns)

        rows_attacked_idx = np.unique(np.concatenate([attack['rows_known'] for attack in attacks]))
        return self.df_original[self.df_original.index.map(lambda x: True if x in rows_attacked_idx else False)]
