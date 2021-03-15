from typing import Dict, Iterator, MutableMapping, Optional, Sequence, cast

import pandas as pd

from .base import Model
from .exceptions import ModelNotFittedError
from ..metadata_new import DataFrameMeta


class DataFrameModel(Model[DataFrameMeta], MutableMapping[str, Model]):
    def __init__(self, meta: DataFrameMeta, models: Optional[Sequence[Model]] = None):
        super().__init__(meta=meta.copy())

        if models is None:
            from .factory import ModelBuilder
            mb = ModelBuilder()
            models = []

            for name, child in meta.items():
                if name in meta.annotations:
                    models.append(mb._from_annotation(child))
                else:
                    models.append(mb(meta))

        self._children = {model.name: model for model in models}

    def fit(self, df: pd.DataFrame) -> 'DataFrameModel':
        super().fit(df=df)

        with self._meta.unfold(df=df):
            for model in self.children:
                model.fit(df=df)

        return self

    def sample(self, n: int, produce_nans: bool = False, conditions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if not self._fitted:
            raise ModelNotFittedError
        dfs = []

        for model in self.children:
            dfs.append(model.sample(n=n, produce_nans=produce_nans, conditions=conditions))

        return pd.concat(dfs, axis=1)

    @property
    def children(self) -> Sequence[Model]:
        """Return the children of this DataFrameModel."""
        return [child for child in self.values()]

    @children.setter
    def children(self, children: Sequence[Model]) -> None:
        self._children = {child.name: child for child in children}

    def __getitem__(self, k: str) -> Model:
        return self._children[k]

    def __iter__(self) -> Iterator[str]:
        for key in self._children:
            yield key

    def __len__(self) -> int:
        return len(self._children)

    def __setitem__(self, k: str, v: Model) -> None:
        self._children[k] = v
        self._meta[k] = v.meta

    def __delitem__(self, k: str) -> None:
        del self._children[k]
        del self._meta[k]

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "children": {m.name: m.to_dict() for m in self.children}
        })
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> 'DataFrameModel':
        df_meta = DataFrameMeta.from_dict(cast(Dict[str, object], d["meta"]))
        models: Sequence[Model] = [Model.from_dict_with_class_name(child_d) for child_d in cast(Dict[str, Dict[str, object]], d["children"]).values()]
        df_model = DataFrameModel(meta=df_meta, models=models)
        df_model._fitted = cast(bool, d["fitted"])

        return df_model
