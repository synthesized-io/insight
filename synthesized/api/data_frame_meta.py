from typing import Optional

from ..metadata import DataFrameMeta as _DataFrameMeta


class DataFrameMeta:

    def __init__(self):
        self._df_meta: Optional[_DataFrameMeta] = None
