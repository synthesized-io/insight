import pickle
from base64 import b64decode, b64encode
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .base import BagOfTransformers, SequentialTransformer, Transformer
from .child import (CategoricalTransformer, DateTransformer, DropConstantColumnTransformer, DTypeTransformer,
                    NanTransformer, QuantileTransformer)
from .exceptions import UnsupportedMetaError
from ..config import MetaTransformerConfig
from ..metadata_new import ContinuousModel, DataFrameMeta, DiscreteModel, Meta, Nominal
from ..metadata_new.base.value_meta import AType, NType


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
                transformers.append(self._from_meta(m))

            return DataFrameTransformer(meta=meta, name=meta.name, transformers=transformers)
        else:
            return self._from_meta(meta)

    def _from_meta(self, meta: Meta) -> Transformer:

        if not isinstance(meta, Nominal):
            raise UnsupportedMetaError(f"{meta.__class__.__name__} has no associated Transformer")

        if isinstance(meta, ContinuousModel):
            transformers = self._from_continuous(meta)

        elif isinstance(meta, DiscreteModel):
            transformers = self._from_discrete(meta)

        assert len(transformers) > 0
        if len(transformers) > 1:
            transformer: Transformer = SequentialTransformer(f'{meta.name}', transformers=transformers)
        else:
            transformer = transformers[0]

        return transformer

    def _from_continuous(self, model: ContinuousModel[AType]) -> List[Transformer]:
        transformers: List[Transformer] = []

        if model.dtype == 'M8[D]':
            transformers.append(DateTransformer.from_meta(model, config=self.config.date_transformer_config))

            if model.nan_freq:
                transformers.append(NanTransformer.from_meta(model))

        elif model.dtype in ['u8', 'f8', 'i8']:
            transformers.append(DTypeTransformer.from_meta(model))

            if model.nan_freq:  # NanTransformer must be here as DTypeTransforemr may produce NaNs
                transformers.append(NanTransformer.from_meta(model))

            transformers.append(QuantileTransformer.from_meta(model, config=self.config.quantile_transformer_config))

        return transformers

    def _from_discrete(self, model: DiscreteModel[NType]) -> List[Transformer]:
        transformers: List[Transformer] = []

        if model.nan_freq:
            transformers.append(NanTransformer.from_meta(model))

        if model.categories is not None and len(model.categories) == 0:
            transformers.append(DropConstantColumnTransformer(f'{model.name}', constant_value=np.nan))
        elif model.categories is not None and len(model.categories) == 1:
            transformers.append(DropConstantColumnTransformer.from_meta(model))
        else:
            transformers.append(CategoricalTransformer.from_meta(model))

        return transformers
