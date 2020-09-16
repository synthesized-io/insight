"""Utilities that help you create Value objects."""
from typing import Dict, Any, Optional, Union, List
import logging

from .associated_categorical import AssociatedCategoricalValue
from .categorical import CategoricalValue
from .continuous import ContinuousValue
from .date import DateValue
from .decomposed_continuous import DecomposedContinuousValue
from .identifier import IdentifierValue
from .rule import RuleValue
from .nan import NanValue
from .value import Value
from ...config import ValueFactoryConfig
from ...metadata import DataFrameMeta, ValueMeta
from ...metadata import CategoricalMeta, ContinuousMeta, DecomposedContinuousMeta, NanMeta, DateMeta, AddressMeta, \
    CompoundAddressMeta, BankNumberMeta, PersonMeta, RuleMeta, IdentifierMeta

logger = logging.getLogger(__name__)


class ValueFactory:
    """A Mix-In that you extend to be able to create various values."""
    def __init__(self, df_meta: DataFrameMeta, name: str = 'value_factory', conditions: List[str] = None,
                 config: ValueFactoryConfig = ValueFactoryConfig()):

        """Init ValueFactory."""
        self.name = name
        self.condition_columns = conditions or list()
        self.config = config

        self.capacity = config.capacity

        # Values
        self.columns = list(df_meta.columns)
        self.values: List[Value] = list()
        self.conditions: List[Value] = list()
        self.identifier_value: Optional[Value] = None

        self.create_values(df_meta)

    def create_values(self, df_meta: DataFrameMeta):
        if df_meta.association_meta is not None:
            associated_metas = [v.name for v in df_meta.association_meta.values]
        else:
            associated_metas = []

        associated_values = []

        for value_meta in df_meta.values:
            value = self.create_value(value_meta)
            if value is not None:
                if value.name in associated_metas:
                    if isinstance(value, CategoricalValue):
                        associated_values.append(value)
                    else:
                        raise ValueError(f'Associated value {value.name} not CategoricalValue.')
                elif value.name in self.condition_columns:
                    assert value_meta.learned_input_columns() == value_meta.learned_output_columns()
                    self.conditions.append(value)
                else:
                    self.values.append(value)

        if len(associated_values) > 0 and df_meta.association_meta is not None:
            self.values.append(AssociatedCategoricalValue(
                values=associated_values, associations=df_meta.association_meta.associations
            ))

        if df_meta.id_value is not None:
            self.identifier_value = self.create_value(df_meta.id_value)

    def create_value(self, vm: Union[ValueMeta, None]) -> Optional[Value]:
        if isinstance(vm, CategoricalMeta):
            if vm.num_categories is None:
                raise ValueError
            return CategoricalValue(
                vm.name, num_categories=vm.num_categories, similarity_based=vm.similarity_based,
                nans_valid=vm.nans_valid, config=self.config.categorical_config
            )
        elif isinstance(vm, DateMeta):
            return DateValue(
                vm.name, categorical_config=self.config.categorical_config,
                continuous_config=self.config.continuous_config
            )
        elif isinstance(vm, NanMeta):
            value = self.create_value(vm.value)
            if value is None:
                raise ValueError
            return NanValue(vm.name, value=value, config=self.config.nan_config)
        elif isinstance(vm, ContinuousMeta):
            return ContinuousValue(
                vm.name, config=self.config.continuous_config
            )
        elif isinstance(vm, DecomposedContinuousMeta):
            return DecomposedContinuousValue(
                vm.name, config=self.config.decomposed_continuous_config
            )
        elif isinstance(vm, IdentifierMeta):
            return IdentifierValue(vm.name, num_identifiers=vm.num_identifiers, config=self.config.identifier_config)
        elif isinstance(vm, AddressMeta):
            if isinstance(vm.postcode, CategoricalMeta):
                if vm.postcode.num_categories is None:
                    raise ValueError
                return CategoricalValue(
                    vm.name, num_categories=vm.postcode.num_categories, similarity_based=vm.postcode.similarity_based,
                    nans_valid=vm.postcode.nans_valid, config=self.config.categorical_config
                )
        elif isinstance(vm, CompoundAddressMeta):
            return self.create_value(vm.postcode)
        elif isinstance(vm, PersonMeta):
            if isinstance(vm.gender, CategoricalMeta):
                if vm.gender.num_categories is None:
                    raise ValueError
                return CategoricalValue(
                    vm.name, num_categories=vm.gender.num_categories, similarity_based=vm.gender.similarity_based,
                    nans_valid=vm.gender.nans_valid, config=self.config.categorical_config
                )
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

    @staticmethod
    def from_dict(variables: dict):
        vf = ValueFactory.__new__(ValueFactory)
        vf.name = 'value_factory'
        vf.set_variables(variables)

        return vf
