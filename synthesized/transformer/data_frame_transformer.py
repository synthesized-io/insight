import pickle
from base64 import b64decode, b64encode
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .base import BagOfTransformers, SequentialTransformer, Transformer
from .child import NanTransformer
from .exceptions import UnsupportedMetaError
from ..config import MetaTransformerConfig
from ..metadata_new import DataFrameMeta, Meta, Nominal


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

    def transform(self, df: pd.DataFrame, inplace: bool = False,
                  **kwargs) -> pd.DataFrame:
        """
        Transform the data frame.

        Args:
            df: The data frame to transform.
            inplace: Whether to transform a copy or inplace.

        Returns:
            The transformed data frame.
        """
        if not inplace:
            df = df.copy()

        for col_name in df.columns:
            self.in_dtypes[col_name] = str(df[col_name].dtype)

        return super().transform(df, **kwargs)

    def inverse_transform(self, df: pd.DataFrame, inplace: bool = False, **kwargs) -> pd.DataFrame:
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

        df = super().inverse_transform(df, **kwargs)
        self.set_dtypes(df)
        return df

    def set_dtypes(self, df: pd.DataFrame) -> None:
        for col_name, col_dtype in self.in_dtypes.items():
            if str(df[col_name].dtype) != str(col_dtype):
                df.loc[:, col_name] = df.loc[:, col_name].astype(col_dtype, errors='ignore')

    @classmethod
    def from_meta(cls, meta: DataFrameMeta) -> 'DataFrameTransformer':
        obj: 'DataFrameTransformer' = TransformerFactory().create_transformers(meta)  # type: ignore
        return obj

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> 'DataFrameTransformer':
        return pickle.loads(b64decode(d['pickle'].encode('utf-8')))

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            name=self.name,
            pickle=b64encode(pickle.dumps(self)).decode('utf-8')
        )


class TransformerFactory:
    """
    Factory class to instantiate Transformers.

    Uses a MetaTransformerConfig to map Transformers, or sequence of Transformers to Meta classes.
    """
    def __init__(self, transformer_config: Optional[MetaTransformerConfig] = None):

        if transformer_config is None:
            self.config = MetaTransformerConfig()
        else:
            self.config = transformer_config

    def create_transformers(self, meta: Union[Nominal, DataFrameMeta]) -> Union[DataFrameTransformer, Transformer]:
        """
        Instantiate Transformers from a Meta instance.

        Args:
            meta: DataFrameMeta or Nominal.

        Returns:
            A Transformer instance.
        """
        if isinstance(meta, DataFrameMeta):
            transformers: List[Transformer] = []
            for m in meta.children:
                if isinstance(m, Nominal) and m.nan_freq is not None and m.nan_freq > 0:
                    transformers.append(
                        SequentialTransformer(m.name, transformers=[NanTransformer.from_meta(m), self._from_meta(m)])
                    )
                else:
                    transformers.append(self._from_meta(m))

            return DataFrameTransformer(meta=meta, name=meta.name, transformers=transformers)
        else:
            return self._from_meta(meta)

    def _from_meta(self, meta: Meta) -> Transformer:

        try:
            transformer_class_name = getattr(self.config, meta.__class__.__name__)
        except KeyError:
            raise UnsupportedMetaError(f"{meta.__class__.__name__} has no associated Transformer")

        if isinstance(transformer_class_name, list):
            transformers = []
            for name in transformer_class_name:
                transformers.append(Transformer.from_name_and_meta(name, meta))
            return SequentialTransformer(name=meta.name, transformers=transformers)
        else:
            return Transformer.from_name_and_meta(transformer_class_name, meta)
