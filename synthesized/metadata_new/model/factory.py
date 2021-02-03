from typing import Any, Dict, Optional, Union, cast

import numpy as np

from . import Histogram, KernelDensityEstimate
from ..base import Affine, Meta, Model, Nominal, ValueMeta
from ..data_frame_meta import DataFrameMeta
from ...config import ModelFactoryConfig


class ModelFactory():
    """Factory class to create Model instances from Meta."""
    def __init__(self, config: Optional[ModelFactoryConfig] = None):
        if config is None:
            self.config = ModelFactoryConfig()
        else:
            self.config = config

    def create_model(self, meta: Meta) -> Union[Model, Dict[str, Model]]:
        """
        Create a KernelDensityEstimate for an Affine Meta, or a Histogram for Nominal. If the number
        of unique categories is smaller than the maximum of of min_num_unique, sqrt(num_rows), or
        categorical_threshold * log(num_rows) then a Histogram is returned.

        Args:
            x: ValueMeta or DataFrameMeta instance.
            num_rows: Optional; number of rows in the column.

        Returns:
            Single Model if x is a ValueMeta or a dictionary mapping column name to model instance if a DataFrameMeta.
        """

        if isinstance(meta, ValueMeta):
            return self._from_value_meta(meta)

        elif isinstance(meta, DataFrameMeta):
            return self._from_dataframe_meta(meta)

        else:
            raise TypeError(f"Cannot create Model from {type(meta)}")

    def _from_value_meta(self, meta: ValueMeta[Any]) -> Model:

        if isinstance(meta, Affine):
            n_unique = len(meta.categories) if meta.categories else 0
            if (meta.categories is None) or\
               (meta.num_rows and (n_unique > max(self.config.min_num_unique,
                                                  self.config.categorical_threshold_log_multiplier * np.log(meta.num_rows)
                                                  ))):
                return KernelDensityEstimate.from_meta(meta)

        if isinstance(meta, Nominal):
            return Histogram.from_meta(meta)

        else:
            raise TypeError(f"Cannot create Model from {type(meta)}")

    def _from_dataframe_meta(self, meta: DataFrameMeta) -> Dict[str, Model]:
        models: Dict[str, Model] = {}
        for name, value_meta in meta.items():
            value_meta = cast(ValueMeta, value_meta)
            model = self._from_value_meta(value_meta)
            models[name] = model
        return models
