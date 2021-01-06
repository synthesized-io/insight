from typing import cast, Any, Optional, Union, Dict

import pandas as pd
import numpy as np

from . import Histogram
from ..base import Nominal, Model, ValueMeta
from ..meta_builder import MetaFactory
from ..data_frame_meta import DataFrameMeta
from ...config import ModelFactoryConfig


class ModelFactory():
    """Factory class to create fitted Model instances from pd.Series and pd.DataFrame objects."""
    def __init__(self, config: Optional[ModelFactoryConfig] = None):
        if config is None:
            self.config = ModelFactoryConfig()
        else:
            self.config = config

    def create_model(self, x: Union[pd.Series, pd.DataFrame], meta: Optional[Union[ValueMeta[Any], DataFrameMeta]] = None,
                     force_discrete: bool = True) -> Union[Model, Dict[str, Model]]:
        """
        Create a Model instance from a Series, or a collection of Models from a DataFrame.

        The mapping of ValueMeta -> Model is specified in ModelFactoryConfig. If Model is None,
        the ValueMeta is assumed to reperesent a continuous value. This mapping is overriden
        by the min_num_unique and the categorical_threshold_multiplier parameters of ModelFactoryConfig.

        Each Model is instantiated and then Model.fit is called on the corresponding data.

        Args:
            x: Series or DataFrame to model.
            meta: ValueMeta or DataFrameMeta instance for the given Series or DataFrame. Defaults to None.
            force_discrete: Flag to return DiscreteModel if the number of unique values is below the
            threshold specified in the config.

        Returns:
            Dictionary mapping column name to model instance.
        """

        if meta is None:
            meta = MetaFactory().create_meta(x)

        if isinstance(x, pd.Series):
            meta = cast(Nominal[Any], meta)
            return self._from_series(x, meta, force_discrete)

        elif isinstance(x, pd.DataFrame):
            meta = cast(DataFrameMeta, meta)
            return self._from_df(x, meta, force_discrete)

        else:
            raise TypeError(f"Cannot create meta from {type(x)}")

    def _from_series(self, sr: pd.Series, meta: Nominal[Any], force_discrete: bool) -> Model:

        if meta.__class__.__name__ not in self.config.meta_model_mapping:
            cls = Histogram

        else:
            n_rows = len(sr)
            n_unique = sr.nunique()
            if force_discrete and (n_unique <= np.sqrt(n_rows)
               or n_unique <= max(self.config.min_num_unique, self.config.categorical_threshold_log_multiplier * np.log(len(sr)))):
                cls = Histogram
            else:
                cls = Model._model_registry[self.config.meta_model_mapping[meta.__class__.__name__]]  # type: ignore
        return cls.from_meta(meta).fit(sr.to_frame())

    def _from_df(self, df: pd.DataFrame, meta: DataFrameMeta, force_discrete: bool) -> Dict[str, Model]:
        models: Dict[str, Model] = {}
        for col in df.columns:
            model = self._from_series(df[col], meta[col], force_discrete)  # type: ignore
            models[col] = model
        return models
