"""Utilities that help you create Value objects."""
import enum
from typing import Dict, Any, Optional, Union, Iterable, List, Set, Tuple
import logging

from dataclasses import dataclass, fields, asdict

from .address import AddressValue
from .associated_categorical import AssociatedCategoricalValue
from .bank_number import BankNumberValue
from .categorical import CategoricalValue, CategoricalConfig
from .compound_address import CompoundAddressValue
from .continuous import ContinuousValue, ContinuousConfig
from .date import DateValue, DateConfig
from .decomposed_continuous import DecomposedContinuousValue
from .identifier import IdentifierValue, IdentifierConfig
from .nan import NanValue, NanConfig
from .person import PersonValue
from .value import Value
from ..metadata import DataPanel


logger = logging.getLogger(__name__)


class TypeOverride(enum.Enum):
    ID = 'ID'
    DATE = 'DATE'
    CATEGORICAL = 'CATEGORICAL'
    CONTINUOUS = 'CONTINUOUS'
    ENUMERATION = 'ENUMERATION'


@dataclass
class ValueFactoryConfig(ContinuousConfig, CategoricalConfig, DateConfig, NanConfig, IdentifierConfig):
    capacity: int = 128
    decompose_continuous_values: bool = False
    produce_nans: bool = False

    @property
    def value_factory_config(self):
        return ValueFactoryConfig(**{f.name: self.__getattribute__(f.name) for f in fields(ValueFactoryConfig)})


class ValueFactory:
    """A Mix-In that you extend to be able to create various values."""
    def __init__(self, data_panel: DataPanel, name: str = 'value_factory',
                 config: ValueFactoryConfig = ValueFactoryConfig()):

        """Init ValueFactory."""
        self.name = name

        self.categorical_kwargs: Dict[str, Any] = asdict(config.categorical_config)
        self.continuous_kwargs: Dict[str, Any] = asdict(config.continuous_config)
        self.nan_kwargs: Dict[str, Any] = asdict(config.nan_config)
        self.date_kwargs: Dict[str, Any] = asdict(config.date_config)
        self.identifier_kwargs: Dict[str, Any] = asdict(config.identifier_config)

        self.capacity = config.capacity
        self.decompose_continuous_values = config.decompose_continuous_values
        self.produce_nans = config.produce_nans

        # Values
        self.columns = list(data_panel.columns)
        self.values: List[Value] = list()
        self.conditions: List[Value] = list()

        self.create_values(data_panel)

    def create_values(self, data_panel):
        for value_meta in data_panel.values:
            value = self.create_value(value_meta)
            if value is not None:
                self.values.append(value)

    def create_identifier(self, name: str) -> IdentifierValue:
        """Create IdentifierValue."""
        return IdentifierValue(name=name, **self.identifier_kwargs)

    def create_categorical(self, name: str, **kwargs) -> CategoricalValue:
        """Create CategoricalValue."""
        categorical_kwargs = dict(self.categorical_kwargs)
        categorical_kwargs.update(kwargs)
        return CategoricalValue(name=name, **categorical_kwargs)

    def create_continuous(self, name: str, **kwargs) -> Union[ContinuousValue, DecomposedContinuousValue]:
        """Create ContinuousValue."""
        continuous_kwargs = dict(self.continuous_kwargs)
        continuous_kwargs.update(kwargs)
        if self.decompose_continuous_values:
            return DecomposedContinuousValue(name=name, identifier=self.identifier_value, **continuous_kwargs)
        else:
            return ContinuousValue(name=name, **continuous_kwargs)

    def create_date(self, name: str) -> DateValue:
        """Create DateValue."""
        return DateValue(
            name=name, categorical_kwargs=self.categorical_kwargs, continuous_kwargs=self.continuous_kwargs,
            **self.date_kwargs
        )

    def create_nan(self, name: str, value: Value, produce_nans: bool) -> NanValue:
        """Create NanValue."""
        nan_kwargs = dict(self.nan_kwargs)
        nan_kwargs['produce_nans'] = produce_nans
        return NanValue(name=name, value=value, **nan_kwargs)

    def create_person(self, i: int) -> PersonValue:
        """Create PersonValue."""
        return PersonValue(
            name='person_{}'.format(i),
            title_label=self.title_label[i] if self.title_label else None,
            gender_label=self.gender_label[i] if self.gender_label else None,
            categorical_kwargs=self.categorical_kwargs
        )

    def create_bank(self, i: int) -> BankNumberValue:
        """Create BankNumberValue."""
        return BankNumberValue(
            name='bank_{}'.format(i)
        )

    def create_compound_address(self) -> CompoundAddressValue:
        """Create CompoundAddressValue."""
        return CompoundAddressValue(name='address', address_label=self.address_label)

    def create_address(self, i: int) -> AddressValue:
        """Create AddressValue."""
        return AddressValue(
            name='address_{}'.format(i), fake=fake,
            postcode_label=self.postcode_label[i] if self.postcode_label else None,
            categorical_kwargs=self.categorical_kwargs
        )

    def create_value(self, value_meta) -> Optional[Value]:
        # TODO: valuemeta -> value logic.
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
            column_aliases=self.column_aliases,
            id_index=self.id_index,
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
