from typing import Dict, Optional, MutableMapping, List

import pandas as pd

from .base import Meta


class DataFrameMeta(Meta, MutableMapping[str, 'Meta']):
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
            column_aliases: Optional[Dict[str, str]] = None, num_columns: Optional[int] = None,
            num_rows: Optional[int] = None
    ):
        super().__init__(name=name)
        self.id_index = id_index
        self.time_index = time_index
        self.column_aliases = column_aliases if column_aliases is not None else {}
        self.num_columns = num_columns
        self.num_rows = num_rows

    def extract(self, df: pd.DataFrame) -> 'DataFrameMeta':
        super().extract(df)
        self.num_columns = len(df.columns)
        self.num_rows = len(df)
        return self

    def __setitem__(self, k: str, v: 'Meta') -> None:
        self._children[k] = v

    def __delitem__(self, k: str) -> None:
        del self._children[k]

    @property
    def column_meta(self) -> Dict[str, Meta]:
        """Get column <-> ValueMeta mapping."""
        col_meta = {}
        for child in self.children:
            col_meta[child.name] = child
        return col_meta

    @property
    def columns(self) -> List:
        return list(self.column_meta.keys())

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "id_index": self.id_index,
            "time_index": self.time_index,
            "column_aliases": self.column_aliases,
            "num_columns": self.num_columns,
            "num_rows": self.num_rows
        })

        return d
