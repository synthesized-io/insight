"""Utilities that help you create Value objects."""
import enum
from typing import Dict, Any, Optional, Union, List
import logging

from dataclasses import dataclass, fields

from .categorical import CategoricalValue, CategoricalConfig
from .continuous import ContinuousValue
from .date import DateValue
from .decomposed_continuous import DecomposedContinuousValue, DecomposedContinuousConfig
from .identifier import IdentifierConfig
from .rule import RuleValue
from .nan import NanValue, NanConfig
from .value import Value
from ...metadata import DataPanel, ValueMeta
from ...metadata import CategoricalMeta, ContinuousMeta, DecomposedContinuousMeta, NanMeta, DateMeta, AddressMeta, \
    CompoundAddressMeta, BankNumberMeta, PersonMeta, RuleMeta


logger = logging.getLogger(__name__)


class TypeOverride(enum.Enum):
    ID = 'ID'
    DATE = 'DATE'
    CATEGORICAL = 'CATEGORICAL'
    CONTINUOUS = 'CONTINUOUS'
    ENUMERATION = 'ENUMERATION'


@dataclass
class ValueFactoryConfig(CategoricalConfig, NanConfig, IdentifierConfig, DecomposedContinuousConfig):
    capacity: int = 128
    produce_nans: bool = False

    @property
    def value_factory_config(self):
        return ValueFactoryConfig(**{f.name: self.__getattribute__(f.name) for f in fields(ValueFactoryConfig)})


class ValueFactory:
    """A Mix-In that you extend to be able to create various values."""
    def __init__(self, data_panel: DataPanel, name: str = 'value_factory', conditions: List[str] = None,
                 config: ValueFactoryConfig = ValueFactoryConfig()):

        """Init ValueFactory."""
        self.name = name
        self.condition_columns = conditions or list()
        self.config = config

        self.capacity = config.capacity
        self.produce_nans = config.produce_nans

        # Values
        self.columns = list(data_panel.columns)
        self.values: List[Value] = list()
        self.conditions: List[Value] = list()
        self.identifier_value: Optional[Value] = None

        self.create_values(data_panel)

    def create_values(self, data_panel):
        for value_meta in data_panel.values:
            value = self.create_value(value_meta)
            print(value_meta.name, value, value.name if value is not None else 'None')
            if value is not None:
                if value.name in self.condition_columns:
                    assert value_meta.learned_input_columns() == value_meta.learned_output_columns()
                    self.conditions.append(value)
                else:
                    self.values.append(value)

    def create_value(self, vm: Union[ValueMeta, None]) -> Optional[Value]:
        if isinstance(vm, CategoricalMeta):
            if vm.num_categories is None:
                raise ValueError
            return CategoricalValue(
                vm.name, num_categories=vm.num_categories, similarity_based=vm.similarity_based,
                nans_valid=vm.nans_valid, produce_nans=self.produce_nans, config=self.config.categorical_config
            )
        elif isinstance(vm, ContinuousMeta):
            return ContinuousValue(
                vm.name, config=self.config.continuous_config
            )
        elif isinstance(vm, DecomposedContinuousMeta):
            return DecomposedContinuousValue(
                vm.name, config=self.config.decomposed_continuous_config
            )
        elif isinstance(vm, NanMeta):
            value = self.create_value(vm.value)
            if value is None:
                raise ValueError
            return NanValue(
                vm.name, value=value, config=self.config.nan_config
            )
        elif isinstance(vm, DateMeta):
            return DateValue(
                vm.name, categorical_config=self.config.categorical_config,
                continuous_config=self.config.continuous_config
            )
        elif isinstance(vm, AddressMeta):
            if vm.fake is False:
                return self.create_value(vm.postcode)
        elif isinstance(vm, CompoundAddressMeta):
            return self.create_value(vm.postcode)
        elif isinstance(vm, PersonMeta):
            if vm.gender is not None:
                return self.create_value(vm.gender)
        elif isinstance(vm, BankNumberMeta):
            # TODO: create BankNumberMeta logic
            return None
        elif isinstance(vm, RuleMeta):
            values: List[Value] = list()
            for meta in vm.values:
                v = self.create_value(meta)
                if v is None:
                    raise ValueError
                else:
                    values.append(v)
            return RuleValue(
                name=vm.name, values=values, num_learned=vm.num_learned
            )

        return None

    @property
    def all_values(self):
        if self.identifier_value:
            return self.values + self.conditions + [self.identifier_value]
        else:
            return self.values + self.conditions

    def get_values(self) -> List[Value]:
        return self.values

    def get_conditions(self) -> List[Value]:
        return self.conditions

    def get_column_names(self) -> List[str]:
        columns = [
            name for value in self.all_values
            for name in value.learned_output_columns()
        ]
        return columns

    def get_variables(self) -> Dict[str, Any]:
        variables: Dict[str, Any] = dict(
            name=self.name,
            columns=self.columns,
            identifier_value=self.identifier_value.get_variables() if self.identifier_value else None
        )

        variables['num_values'] = len(self.values)
        for i, value in enumerate(self.values):
            variables['value_{}'.format(i)] = value.get_variables()

        variables['num_conditions'] = len(self.conditions)
        for i, condition in enumerate(self.conditions):
            variables['condition_{}'.format(i)] = condition.get_variables()

        return variables

    def set_variables(self, variables: Dict[str, Any]):
        assert self.name == variables['name']

        self.columns = variables['columns']

        self.identifier_value = Value.set_variables(variables['identifier_value']) \
            if variables['identifier_value'] is not None else None

        self.values = []
        for i in range(variables['num_values']):
            self.values.append(Value.set_variables(variables['value_{}'.format(i)]))

        self.conditions = []
        for i in range(variables['num_conditions']):
            self.values.append(Value.set_variables(variables['condition_{}'.format(i)]))


class ValueFactoryWrapper(ValueFactory):
    def __init__(self, name: str, variables: Dict[str, Any]):
        self.name = name
        self.set_variables(variables)
