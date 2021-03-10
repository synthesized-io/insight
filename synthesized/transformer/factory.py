from typing import Dict, List, Optional, Union, cast

import numpy as np

from .base import SequentialTransformer, Transformer
from .child import (CategoricalTransformer, DateTransformer, DropConstantColumnTransformer, DTypeTransformer,
                    NanTransformer, QuantileTransformer)
from .child.gender import GenderTransformer
from .child.postcode import PostcodeTransformer
from .data_frame import DataFrameTransformer
from .exceptions import UnsupportedMetaError
from ..config import MetaTransformerConfig
from ..metadata_new import ContinuousModel, DataFrameMeta, DiscreteModel, Meta, Nominal
from ..metadata_new.base.value_meta import AType, NType
from ..metadata_new.model.address import PostcodeModel
from ..metadata_new.model.association import AssociatedHistogram
from ..metadata_new.model.person import GenderModel


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

        elif isinstance(meta, ContinuousModel):
            transformers = self._from_continuous(meta)

        elif isinstance(meta, AssociatedHistogram):
            transformers = self._from_association(meta)

        elif isinstance(meta, DiscreteModel):
            transformers = []

            if isinstance(meta, GenderModel):
                transformers.append(GenderTransformer.from_meta(meta))
            elif isinstance(meta, PostcodeModel):
                transformers.append(PostcodeTransformer.from_meta(meta))

            transformers.extend(self._from_discrete(cast(DiscreteModel, meta)))

        assert len(transformers) > 0
        if len(transformers) > 1:
            transformer: Transformer = SequentialTransformer(f'{meta.name}', transformers=transformers)
        else:
            transformer = transformers[0]

        return transformer

    def _from_continuous(self, model: ContinuousModel[AType]) -> List[Transformer]:
        transformers: List[Transformer] = []

        if model.dtype == 'M8[ns]':
            transformers.append(DateTransformer.from_meta(model, config=self.config.date_transformer_config))

            if model.nan_freq:
                transformers.append(NanTransformer.from_meta(model))

        elif model.dtype in ['u8', 'f8', 'i8']:
            transformers.append(DTypeTransformer.from_meta(model))

            if model.nan_freq:  # NanTransformer must be here as DTypeTransforemr may produce NaNs
                transformers.append(NanTransformer.from_meta(model))

            transformers.append(QuantileTransformer.from_meta(model, config=self.config.quantile_transformer_config))

        return transformers

    def _from_discrete(self, model: DiscreteModel[NType], category_to_idx: Dict[str, int] = None) -> List[Transformer]:
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

    def _from_association(self, model: AssociatedHistogram):
        transformers: List[Transformer] = []

        for name, child_model in model.items():
            transformers += self._from_discrete(child_model, category_to_idx=model.categories_to_idx[name])  # type: ignore

        return transformers
