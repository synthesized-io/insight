from typing import Dict, Optional

import pandas as pd

from .base_meta import Meta


class DataFrameMeta(Meta):
    """
    Meta to describe an arbitrary data frame.

    Each column is described by a derived ValueMeta object.

    Attributes:
        id_index: NotImplemented
        time_index: NotImplemented
        column_aliases: dictionary mapping column names to an alias.
    """
    def __init__(
            self, name: str, id_index: Optional[str] = None, time_index: Optional[str] = None,
            column_aliases: Optional[Dict[str, str]] = None, num_columns: Optional[int] = None
    ):
        super().__init__(name=name)
        self.id_index = id_index
        self.time_index = time_index
        self.column_aliases = column_aliases if column_aliases is not None else {}
        self.num_columns = num_columns

    def extract(self, df: pd.DataFrame) -> 'DataFrameMeta':
        super().extract(df)
        self.num_columns = len(df.columns)
        return self

    @property
    def column_meta(self) -> Dict[str, Meta]:
        """Get column <-> ValueMeta mapping."""
        col_meta = {}
        for child in self.children:
            col_meta[child.name] = child
        return col_meta
