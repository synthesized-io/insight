from typing import Optional

from ..metadata import DataFrameMeta as _DataFrameMeta


class DataFrameMeta:
    """Container for additional metadata describing each column of a dataset."""
    def __init__(self):
        self._df_meta: Optional[_DataFrameMeta] = None
