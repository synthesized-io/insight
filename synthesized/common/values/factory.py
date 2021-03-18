"""Utilities that help you create Value objects."""
import logging
from typing import Dict, List, Optional, Sequence

from .associated_categorical import AssociatedCategoricalValue
from .categorical import CategoricalValue
from .continuous import ContinuousValue
from .dataframe_value import DataFrameValue
from .date import DateValue
from .value import Value
from ...config import ValueFactoryConfig
from ...model import ContinuousModel, DataFrameModel, DiscreteModel, Model
from ...model.models import AssociatedHistogram

logger = logging.getLogger(__name__)


class ValueFactory:
    """
    Factory for creating Values, through being subclassesed by MetaExtractor
    and building from a DataFrame meta.
    Additionally manages the creation of an Associated NaN values

    Attributes:
        name: A string labelling the instance
        config: (dataclass) configuration namespace
        capacity: Integer taken from config denoting capacity of network
        identifier_value: (Unused)

    """
    def __init__(self, name: str = 'value_factory', config: ValueFactoryConfig = ValueFactoryConfig()):

        """Init ValueFactory."""
        self.name = name
        self.config = config

        self.capacity = config.capacity
        self.identifier_value: Optional[Value] = None

    def _build(self, df_meta: DataFrameModel) -> DataFrameValue:
        """
        Loops over entries in the DataFrameMeta and creates a Value for each
        Manages creation of associated value for NaNs
        """
        values: Dict[str, Value] = dict()
        for name, value_meta in df_meta.items():
            nan_values = self._build_nan(value_meta)
            for nan_value in nan_values:
                values[f"{nan_value.name}"] = nan_value

            value = self._create_value(value_meta)

            if value is not None:
                values[name] = value

        return DataFrameValue(name="dataframe_value", values=values)

    def _create_value(self, vm: Model) -> Optional[Value]:
        """Lookup function for determining correct value for given value meta"""
        if isinstance(vm, ContinuousModel) and vm.meta.dtype == 'M8[ns]':
            return DateValue(
                vm.name, categorical_config=self.config.categorical_config,
                continuous_config=self.config.continuous_config,
            )
        elif isinstance(vm, ContinuousModel):
            return ContinuousValue(
                vm.name, config=self.config.continuous_config,
            )
        elif isinstance(vm, AssociatedHistogram):
            values = {}
            for model in vm.values():
                values[model.name] = CategoricalValue(
                    name=model.name, num_categories=len(model.categories),
                    config=self.config.categorical_config
                )
            return AssociatedCategoricalValue(values=values, binding_mask=vm.binding_mask, name=vm.name)
        elif isinstance(vm, DiscreteModel):
            assert vm.categories is not None
            if len(vm.categories) > 1:
                return CategoricalValue(
                    vm.name, num_categories=len(vm.categories),
                    config=self.config.categorical_config,
                )
            else:
                return None
        else:
            raise ValueError("Cannot build Value from given Meta/Model")

    def _build_nan(self, value_meta: Model) -> Sequence[CategoricalValue]:
        """ builds a list of nan_values from a value_meta if needed else returns None"""
        if isinstance(value_meta, AssociatedHistogram):
            nans: List[CategoricalValue] = []
            for vm in value_meta.children:
                nans.extend(self._build_nan(vm))
            return nans
        elif isinstance(value_meta, ContinuousModel) or isinstance(value_meta, DiscreteModel):
            if value_meta.nan_freq is None or value_meta.nan_freq == 0:
                return []

            nan_value = CategoricalValue(
                name=f"{value_meta.name}_nan", num_categories=2,
                config=self.config.categorical_config
            )
            return [nan_value]

        return []


class ValueExtractor(ValueFactory):
    """Extract the DataFrameMeta for a data frame"""
    def __init__(self, name: str = 'value_factory',
                 config: ValueFactoryConfig = ValueFactoryConfig()):
        super().__init__(name, config)

    @staticmethod
    def extract(df_meta: DataFrameModel, name: str = 'data_frame_value',
                config: ValueFactoryConfig = ValueFactoryConfig()) -> DataFrameValue:
        """
        Instantiate and extract the DataFrameValue that provides the value functionality for the whole dataframe.

        Args:
            df_meta: the data frame meta object that describes the properties of the data.
            config: Optional; The configuration parameters to teh ValueFactory.

        Returns:
            A DataFrameValue instance for which all child values have been instantiated.
        """

        factory = ValueExtractor(name, config)
        df_value = factory._build(df_meta)
        return df_value
