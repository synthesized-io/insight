import pickle
from base64 import b64decode, b64encode
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import BagOfTransformers, Transformer
from ..config import MetaTransformerConfig
from ..metadata import DataFrameMeta
from ..model import DataFrameModel


class DataFrameTransformer(BagOfTransformers):
    """
    Transform a data frame.

    This is a SequentialTransformer built from a DataFrameMeta instance.

    Attributes:
        meta: DataFrameMeta instance returned from MetaExtractor.extract
    """
    def __init__(self, meta: DataFrameMeta, name: Optional[str] = 'df',
                 transformers: Optional[List[Transformer]] = None):
        if name is None:
            raise ValueError("name must not be a string, not None")
        super().__init__(name, transformers=transformers)
        self.meta = meta
        self.in_dtypes: Dict[str, str] = dict()

    def transform(self, df: pd.DataFrame, max_workers: Optional[int] = None, inplace: bool = False,
                  **kwargs) -> pd.DataFrame:
        """
        Transform the data frame.

        Args:
            df: The data frame to transform.
            max_workers: Number of workers user to parallelize transformation, 1 for sequential transformation.
            inplace: Whether to transform a copy or inplace.

        Returns:
            The transformed data frame.
        """
        if not inplace:
            df = df.copy()

        for col_name in df.columns:
            self.in_dtypes[col_name] = str(df[col_name].dtype)

        return super().transform(df, max_workers=max_workers, **kwargs)

    def inverse_transform(self, df: pd.DataFrame, max_workers: Optional[int] = None,
                          inplace: bool = False, **kwargs) -> pd.DataFrame:
        """
        Inverse transform the data frame.

        Args:
            df: The data frame to transform.
            inplace: Whether to transform a copy or inplace.

        Returns:
            The transformed data frame.
        """
        if not inplace:
            df = df.copy()

        df = super().inverse_transform(df, max_workers=max_workers, **kwargs)
        self.set_dtypes(df)
        return df

    def set_dtypes(self, df: pd.DataFrame) -> None:
        for col_name, col_dtype in self.in_dtypes.items():
            if col_name in df.columns and str(df[col_name].dtype) != str(col_dtype):
                df.loc[:, col_name] = df.loc[:, col_name].astype(col_dtype, errors='ignore')

    @classmethod
    def from_meta(cls, meta: DataFrameModel, config: MetaTransformerConfig = None) -> 'DataFrameTransformer':
        from .factory import TransformerFactory
        obj: 'DataFrameTransformer' = TransformerFactory(config=config).create_transformers(meta)  # type: ignore
        return obj

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> 'DataFrameTransformer':
        return pickle.loads(b64decode(d['pickle'].encode('utf-8')))

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            name=self.name,
            pickle=b64encode(pickle.dumps(self)).decode('utf-8')
        )
