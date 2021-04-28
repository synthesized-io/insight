from typing import List, Optional, Union

import numpy as np

from .base import ContinuousModel, DiscreteModel
from .data_frame_model import DataFrameModel
from .models import (AddressModel, AssociatedHistogram, BankModel, FormattedStringModel, Histogram,
                     KernelDensityEstimate, PersonModel)
from ..config import ModelBuilderConfig
from ..metadata import Affine, DataFrameMeta, Nominal
from ..metadata.value import Address, AssociatedCategorical, Bank, FormattedString, Person

DisContModel = Union[DiscreteModel, ContinuousModel]


class ModelFactory:
    """Factory class to create DataFrameMetas of Model instances from DataFrameMetas."""

    def __init__(self, config: ModelBuilderConfig = None):
        self._builder = ModelBuilder(config)

    def __call__(self, df_meta: DataFrameMeta, type_overrides: Optional[List[DisContModel]] = None) -> DataFrameModel:
        """
        Public method for creating a model container from a DataFrameMeta
        Args:
            meta: DataFrameMeta instance.
            type_overrides: List containing the models that override the default mappings
        Returns:
            df_model: a DataFrameMeta mapping column name to model instance.
        """
        type_override_dict = {m.name: m for m in type_overrides} if type_overrides is not None else {}
        models = []
        for name, meta in df_meta.items():
            if meta.name in type_override_dict:
                models.append(type_override_dict[meta.name])
            elif meta.name in df_meta.annotations:
                models.append(self._builder._from_annotation(meta))
            else:
                models.append(self._builder(meta))

        df_model = DataFrameModel(meta=df_meta, models=models)
        return df_model


class ModelBuilder:
    """ Wraps the logic around how the default mapping of meta -> model takes place """
    def __init__(self, config: ModelBuilderConfig = None):
        if config is None:
            self.config = ModelBuilderConfig()
        else:
            self.config = config

    def __call__(self, meta):
        """
        Creates a KernelDensityEstimate for an Affine Meta, or a Histogram for Nominal. If the number
        of unique categories is smaller than the maximum of of min_num_unique, sqrt(num_rows), or
        categorical_threshold * log(num_rows) then a Histogram is returned.

        Args:
            meta: Meta instance, if not of type at least Nominal, will raise an error

        Returns:
            Model if meta is a valid meta Meta
        """
        if isinstance(meta, Affine):
            n_unique = len(meta.categories) if meta.categories else 0
            if (meta.categories is None) or\
               (meta.num_rows and (n_unique > max(self.config.min_num_unique,
                                                  self.config.categorical_threshold_log_multiplier * np.log(meta.num_rows)
                                                  ))):
                return KernelDensityEstimate(meta)

        if isinstance(meta, Nominal):
            return Histogram(meta)

        if isinstance(meta, AssociatedCategorical):
            return AssociatedHistogram(meta)
        else:
            raise TypeError(f"Cannot create Model from {type(meta)}")

    def _from_annotation(self, meta):
        """
        Create a model from an annotated meta.

        Args:
            meta: Meta instance

        Returns:
            Model if meta is a ValueMeta
        """
        if isinstance(meta, Address):
            return AddressModel(meta, config=self.config.address_model_config)
        elif isinstance(meta, Bank):
            return BankModel(meta)
        elif isinstance(meta, Person):
            return PersonModel(meta, config=self.config.person_model_config)
        elif isinstance(meta, FormattedString):
            return FormattedStringModel(meta)
        else:
            raise ValueError(f"Unable to recognise given annotation meta '{meta}'")
