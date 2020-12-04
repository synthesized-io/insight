from typing import Optional, Dict, cast

from dataclasses import asdict

from .exceptions import UnsupportedMetaError
from .base import Transformer, SequentialTransformer
from .nan import NanTransformer
from ..metadata_new import Meta, DataFrameMeta, Affine
from ..config import MetaExtractorConfig


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
    def __init__(self, transformer_config: Optional[MetaExtractorConfig] = None):

        if transformer_config is None:
            self.config: Dict[str, object] = asdict(MetaExtractorConfig())
        else:
            self.config = asdict(transformer_config)

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
            meta_transformer = self.config[meta.__class__.__name__]
        except KeyError:
            raise UnsupportedMetaError(f"{meta.__class__.__name__} has no associated Transformer")

        if isinstance(meta_transformer, list):
            transformer = SequentialTransformer(f'Sequential({meta.name})')
            for t in meta_transformer:
                kwargs = self._get_transformer_kwargs(meta, t)
                transformer.add_transformer(t.from_meta(meta, **kwargs))
        else:
            transformer = meta_transformer.from_meta(meta, **self._get_transformer_kwargs(meta, meta_transformer))  #type: ignore

        if isinstance(meta, Affine) and meta.nan_freq > 0:  # type: ignore
            transformer += NanTransformer.from_meta(meta)

        return transformer  # type: ignore

    def _get_transformer_kwargs(self, meta: Meta, transformer: str) -> Dict[str, object]:
        try:
            return cast(Dict[str, object], self.config[meta.name])
        except KeyError:
            return {}
