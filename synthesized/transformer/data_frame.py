from typing import Optional, Union

import pandas as pd

from .exceptions import UnsupportedMetaError
from .base import Transformer, SequentialTransformer
from .nan import NanTransformer
from ..metadata_new import Meta, DataFrameMeta, Nominal
from ..config import MetaTransformerConfig


class DataFrameTransformer(SequentialTransformer):
    """
    Transform a data frame.

    This is a SequentialTransformer built from a DataFrameMeta instance.

    Attributes:
        meta: DataFrameMeta instance returned from MetaExtractor.extract
    """
    def __init__(self, meta: DataFrameMeta, name: Optional[str] = 'df'):
        if name is None:
            raise ValueError("name must not be a string, not None")
        super().__init__(name)
        self.meta = meta

    def transform(self, df: pd.DataFrame, inplace: bool = False, max_workers: Optional[int] = None) -> pd.DataFrame:
        """
        Transform the data frame.

        Args:
            df: The data frame to transform.
            inplace: Whether to transform a copy or inplace.
            max_workers: Number of workers to process in parallel.

        Returns:
            The transformed data frame.
        """
        if not inplace:
            df = df.copy()

        # To do: implement parallel transform
        for transformer in self:
            df = transformer.transform(df)

        return df

    def inverse_transform(self, df: pd.DataFrame, inplace: bool = False, max_workers: Optional[int] = None) -> pd.DataFrame:
        """
        Inverse transform the data frame.

        Args:
            df: The data frame to transform.
            inplace: Whether to transform a copy or inplace.
            max_workers: Number of workers to process in parallel.

        Returns:
            The transformed data frame.
        """
        if not inplace:
            df = df.copy()

        # To do: implement parallel transform
        for transformer in reversed(self):
            df = transformer.inverse_transform(df)

        return df

    @classmethod
    def from_meta(cls, meta: DataFrameMeta) -> 'DataFrameTransformer':
        obj: 'DataFrameTransformer' = TransformerFactory().create_transformers(meta)  # type: ignore
        return obj


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
            transformer = DataFrameTransformer(meta, meta.name)
            for m in meta.children:
                if isinstance(m, Nominal):
                    if m.nan_freq is not None and m.nan_freq > 0:
                        transformer.append(NanTransformer.from_meta(m))

                transformer.append(self._from_meta(m))
        else:
            return self._from_meta(meta)
        return transformer

    def _from_meta(self, meta: Meta) -> Transformer:

        try:
            transformer_class_name = getattr(self.config, meta.__class__.__name__)
        except KeyError:
            raise UnsupportedMetaError(f"{meta.__class__.__name__} has no associated Transformer")

        if isinstance(transformer_class_name, list):
            transformer = SequentialTransformer(f'{meta.name}')
            for name in transformer_class_name:
                t = Transformer._transformer_registry[name]
                transformer.append(t.from_meta(meta))
        else:
            transformer_cls = Transformer._transformer_registry[transformer_class_name]
            transformer = transformer_cls.from_meta(meta)  # type: ignore

        return transformer
