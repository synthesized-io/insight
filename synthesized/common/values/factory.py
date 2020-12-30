"""Utilities that help you create Value objects."""
from typing import Dict, Optional, List
import logging


from .categorical import CategoricalValue
from .continuous import ContinuousValue
from .date import DateValue
from .dataframe_value import DataFrameValue
from .nan import NanValue
from .value import Value
from ...config import ValueFactoryConfig
from ...metadata_new import DataFrameMeta, Meta, Nominal
from ...metadata_new import Bool, IntegerBool, String, OrderedString, Integer, Float, Date

logger = logging.getLogger(__name__)


class ValueFactory:
    """
    Factory for creating Values, through being subclassesed by MetaExtractor
    and building from a DataFrame meta.
    Additionally manages the creation of an Associated NaN values

    Attributes:
        name: A string labelling the instance
        conditions: (Unused) List for denoting conditions
        config: (dataclass) configuration namespace
        capacity: Integer taken from config denoting capacity of network
        identifier_value: (Unused)

    """
    def __init__(self, name: str = 'value_factory', conditions: List[str] = None,
                 config: ValueFactoryConfig = ValueFactoryConfig()):

        """Init ValueFactory."""
        self.name = name
        self.config = config

        self.capacity = config.capacity
        self.conditions: List[Value] = list()
        self.identifier_value: Optional[Value] = None

    def _build(self, df_meta: DataFrameMeta) -> DataFrameValue:
        """
        Loops over entries in the DataFrameMeta and creates a Value for each
        Manages creation of associated value for NaNs
        """
        values: Dict[str, Value] = dict()
        for name, value_meta in df_meta.items():
            if isinstance(value_meta, Nominal):
                nan_value = self._build_nan(name, value_meta)
                if nan_value is not None:
                    values[f"{name}_nan"] = nan_value
                value = self._create_value(value_meta)
            else:
                raise ValueError('Unsupported Building of DataFrameValue with non-nominal metas')

            values[name] = value

        return DataFrameValue(name="dataframe_value", values=values)

    def _create_value(self, vm: Meta, has_nan: bool = False) -> Value:
        """Lookup function for determining correct value for given value meta"""
        if isinstance(vm, (String, OrderedString)):
            assert vm.categories is not None
            return CategoricalValue(
                vm.name, num_categories=len(vm.categories),
                config=self.config.categorical_config,
            )
        elif isinstance(vm, Date):
            return DateValue(
                vm.name, categorical_config=self.config.categorical_config,
                continuous_config=self.config.continuous_config,
            )
        elif isinstance(vm, (Float, Integer)):
            return ContinuousValue(
                vm.name, config=self.config.continuous_config,
            )
        elif isinstance(vm, (Bool, IntegerBool)):
            return CategoricalValue(
                vm.name, num_categories=2,
                config=self.config.categorical_config,
            )
        else:
            raise ValueError("Bad Nominal Value Meta")

    def _build_nan(self, name, value_meta: Nominal) -> Optional[CategoricalValue]:
        """ builds a nan_value from a value_meta if needed else returns None"""
        if value_meta.nan_freq is None or value_meta.nan_freq == 0:
            return None
        nan_value = NanValue(name=f"{name}_nan", config=self.config.categorical_config)
        return nan_value


class ValueExtractor(ValueFactory):
    """Extract the DataFrameMeta for a data frame"""
    def __init__(self, name: str = 'value_factory', conditions: List[str] = None,
                 config: ValueFactoryConfig = ValueFactoryConfig()):
        super().__init__(name, conditions, config)

    @staticmethod
    def extract(df_meta: DataFrameMeta, name: str = 'data_frame_value', conditions: List[str] = None,
                config: ValueFactoryConfig = ValueFactoryConfig()) -> DataFrameValue:
        """
        Instantiate and extract the DataFrameValue that provides the value functionality for the whole dataframe.

        Args:
            df_meta: the data frame meta object that describes the properties of the data.
            config: Optional; The configuration parameters to teh ValueFactory.

        Returns:
            A DataFrameValue instance for which all child values have been instantiated.
        """

        factory = ValueExtractor(name, conditions, config)
        df_value = factory._build(df_meta)
        return df_value
