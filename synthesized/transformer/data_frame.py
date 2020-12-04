from typing import Optional, Dict, cast

from dataclasses import asdict

from .exceptions import UnsupportedMetaError
from .base import Transformer, SequentialTransformer, _transformer_registry
from .nan import NanTransformer
from ..metadata_new import Meta, DataFrameMeta, Affine
from ..config import MetaTransformerConfig


class DataFrameTransformer(SequentialTransformer):
    """
    Transform a data frame.

    This is a SequentialTransform built from a DataFrameMeta instance.

    Attributes:
        meta: DataFrameMeta instance returned from MetaExtractor.extract
    """
    def __init__(self, meta: DataFrameMeta, name: Optional[str] = 'df'):
        if name is None:
            raise ValueError("name must not be a string, not None")
        super().__init__(name)
        self.meta = meta

    @classmethod
    def from_meta(cls, meta: DataFrameMeta) -> 'DataFrameTransformer':
        obj = TransformerFactory().create_transformers(meta)
        return obj


# Todo: We should consider how we could serialize the config class. Maybe each Transformer could have the "class_name"
#       property and we implement something similar to the Meta.STR_TO_META.
class TransformerFactory:
    def __init__(self, transformer_config: Optional[MetaTransformerConfig] = None):

        if transformer_config is None:
            self.config = MetaTransformerConfig()
        else:
            self.config = transformer_config

    def create_transformers(self, meta: Meta) -> Transformer:

        if isinstance(meta, DataFrameMeta):
            transformer = DataFrameTransformer(meta, meta.name)
            for m in meta.children:
                transformer.add_transformer(self._from_meta(m))
        else:
            return self._from_meta(meta)
        return transformer

    def _from_meta(self, meta: Meta) -> Transformer:

        try:
            transformer_class_name = getattr(self.config, meta.__class__.__name__)
        except KeyError:
            raise UnsupportedMetaError(f"{meta.__class__.__name__} has no associated Transformer")

        if isinstance(transformer_class_name, list):
            transformer = SequentialTransformer(f'Sequential({meta.name})')
            for name in transformer_class_name:
                t = _transformer_registry[name]
                transformer.add_transformer(t.from_meta(meta))
        else:
            transformer_cls = _transformer_registry[transformer_class_name]
            transformer = transformer_cls.from_meta(meta)  # type: ignore

        if isinstance(meta, Affine) and meta.nan_freq > 0:  # type: ignore
            transformer += NanTransformer.from_meta(meta)

        return transformer  # type: ignore

    def _get_transformer_kwargs(self, transformer: str) -> Dict[str, object]:
        try:
            return cast(Dict[str, object], asdict(self.config.get_transformer_config(transformer)))
        except KeyError:
            return {}
