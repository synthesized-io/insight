"""Utilities that help you create Value objects."""
import enum
from math import log, sqrt
from typing import Dict, Any, Optional, Union, Iterable, List, Set
from collections import OrderedDict

import numpy as np
import pandas as pd
import tensorflow as tf

from .address import AddressValue
from .categorical import CategoricalValue
from .compound_address import CompoundAddressValue
from .continuous import ContinuousValue
from .date import DateValue
from .enumeration import EnumerationValue
from .identifier import IdentifierValue
from .identify_rules import identify_rules
from .nan import NanValue
from .person import PersonValue
from .sampling import SamplingValue
from .bank_number import BankNumberValue
from .constant import ConstantValue
from .value import Value
from ..module import tensorflow_name_scoped

CATEGORICAL_THRESHOLD_LOG_MULTIPLIER = 2.5
PARSING_NAN_FRACTION_THRESHOLD = 0.25


class TypeOverride(enum.Enum):
    ID = 'ID'
    DATE = 'DATE'
    CATEGORICAL = 'CATEGORICAL'
    CONTINUOUS = 'CONTINUOUS'
    ENUMERATION = 'ENUMERATION'


class ValueFactory(tf.Module):
    """A Mix-In that you extend to be able to create various values."""

    def __init__(
        self, df: pd.DataFrame, capacity: int = 128, weight_decay: float = 1e-5, continuous_weight: float = 1.0,
        categorical_weight: float = 1.0, temperature: float = 1.0, moving_average: bool = True, smoothing: float = 0.1,
        entropy_regularization: float = 0.1, similarity_regularization: float = 0.1,
        name: str = 'value_factory',
        type_overrides: Dict[str, TypeOverride] = None,
        produce_nans_for: Union[bool, Iterable[str], None] = None,
        column_aliases: Dict[str, str] = None, condition_columns: List[str] = None,
        find_rules: Union[str, List[str]] = None,
        # Person
        title_label: str = None, gender_label: str = None, name_label: str = None, firstname_label: str = None,
        lastname_label: str = None, email_label: str = None,
        mobile_number_label: str = None, home_number_label: str = None, work_number_label: str = None,
        # Bank
        bic_label: str = None, sort_code_label: str = None, account_label: str = None,
        # Address
        postcode_label: str = None, county_label: str = None, city_label: str = None,
        district_label: str = None,
        street_label: str = None, house_number_label: str = None, flat_label: str = None,
        house_name_label: str = None,
        address_label: str = None, postcode_regex: str = None,
        # Identifier
        identifier_label: str = None,
    ):

        super(ValueFactory, self).__init__(name=name)
        """Init ValueFactory."""
        categorical_kwargs: Dict[str, Any] = dict()
        continuous_kwargs: Dict[str, Any] = dict()
        nan_kwargs: Dict[str, Any] = dict()
        categorical_kwargs['capacity'] = capacity
        nan_kwargs['capacity'] = capacity
        categorical_kwargs['weight_decay'] = weight_decay
        nan_kwargs['weight_decay'] = weight_decay
        categorical_kwargs['weight'] = categorical_weight
        nan_kwargs['weight'] = categorical_weight
        continuous_kwargs['weight'] = continuous_weight
        categorical_kwargs['temperature'] = temperature
        categorical_kwargs['moving_average'] = moving_average

        self.categorical_kwargs = categorical_kwargs
        self.continuous_kwargs = continuous_kwargs
        self.nan_kwargs = nan_kwargs

        if find_rules is None:
            self.find_rules: Union[str, List[str]] = []
        else:
            self.find_rules = find_rules

        # Values
        self.columns = list(df.columns)
        self.values: List[Value] = list()
        self.conditions: List[Value] = list()

        self.capacity = capacity
        self.weight_decay = weight_decay

        # Person
        self.person_value: Optional[Value] = None
        self.bank_value: Optional[Value] = None
        self.title_label = title_label
        self.gender_label = gender_label
        self.name_label = name_label
        self.firstname_label = firstname_label
        self.lastname_label = lastname_label
        self.email_label = email_label
        self.mobile_number_label = mobile_number_label
        self.home_number_label = home_number_label
        self.work_number_label = work_number_label
        self.bic_label = bic_label
        self.sort_code_label = sort_code_label
        self.account_label = account_label
        # Address
        self.address_value: Optional[Value] = None
        self.postcode_label = postcode_label
        self.county_label = county_label
        self.city_label = city_label
        self.district_label = district_label
        self.street_label = street_label
        self.house_number_label = house_number_label
        self.flat_label = flat_label
        self.house_name_label = house_name_label
        self.address_label = address_label
        self.postcode_regex = postcode_regex
        # Identifier
        self.identifier_value: Optional[Value] = None
        self.identifier_label = identifier_label
        # Date
        self.date_value: Optional[Value] = None

        if type_overrides is None:
            self.type_overrides: Dict[str, TypeOverride] = dict()
        else:
            self.type_overrides = type_overrides

        if isinstance(produce_nans_for, Iterable):
            self.produce_nans_for: Set[str] = set(produce_nans_for)
        elif produce_nans_for:
            self.produce_nans_for = set(df.columns)
        else:
            self.produce_nans_for = set()

        if column_aliases is None:
            self.column_aliases: Dict[str, str] = {}
        else:
            self.column_aliases = column_aliases

        if condition_columns is None:
            self.condition_columns: List[str] = []
        else:
            self.condition_columns = condition_columns

        for name in df.columns:
            # we are skipping aliases
            if name in self.column_aliases:
                continue
            if name in self.type_overrides:
                value = self._apply_type_overrides(df, name)
            else:
                identified_value = self.identify_value(col=df[name], name=name)
                # None means the value has already been detected:
                if identified_value is None:
                    continue
                value = identified_value
            if name in self.condition_columns:
                self.conditions.append(value)
            else:
                self.values.append(value)

        # Automatic extraction of specification parameters
        df = df.copy()
        for value in (self.values + self.conditions):
            value.extract(df=df)

        # Identify deterministic rules
        #  import ipdb; ipdb.set_trace()
        self.values = identify_rules(values=self.values, df=df, tests=self.find_rules)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a preprocessed copy of the input DataFrame"""
        df_copy = df.copy()
        for value in (self.values + self.conditions):
            df_copy = value.preprocess(df=df_copy)

        return df_copy

    def postprocess(self,  df: pd.DataFrame) -> pd.DataFrame:
        """Post-processes the input DataFrame"""
        for value in (self.values + self.conditions):
            df = value.postprocess(df=df)

        # aliases:
        for alias, col in self.column_aliases.items():
            df[alias] = df[col]

        assert len(df.columns) == len(self.columns)
        df = df[self.columns]

        return df

    def preprocess_conditions(self, conditions: Union[pd.DataFrame, None]) -> Union[pd.DataFrame, None]:
        """Returns a preprocessed copy of the input conditions dataframe"""
        if conditions is not None:
            if isinstance(conditions, dict):
                df_conditions = pd.DataFrame.from_dict(
                    {name: np.reshape(condition, (-1,)) for name, condition in conditions.items()}
                )
            else:
                df_conditions = conditions.copy()

            for value in self.conditions:
                df_conditions = value.preprocess(df=df_conditions)
        else:
            df_conditions = None

        return df_conditions

    def get_data_feed_dict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        data = {
            name: df[name].to_numpy() for value in (self.values + self.conditions)
            for name in value.learned_input_columns()
        }
        return data

    def get_conditions_data(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        data = {
            name: df[name].to_numpy() for value in (self.conditions)
            for name in value.learned_input_columns()
        }
        return data

    def get_conditions_feed_dict(self, df_conditions, num_rows):
        feed_dict = dict()

        if (num_rows % 1024) != 0:
            for value in self.conditions:
                for name in value.learned_input_columns():
                    condition = df_conditions[name].values
                    if condition.shape == (1,):
                        feed_dict[name] = np.tile(condition, (num_rows % 1024,))
                    elif condition.shape == (num_rows,):
                        feed_dict[name] = condition[-num_rows % 1024:]
                    else:
                        raise NotImplementedError
        else:
            for value in self.conditions:
                for name in value.learned_input_columns():
                    condition = df_conditions[name].values
                    if condition.shape == (1,):
                        feed_dict[name] = np.tile(condition, (1024,))

        return feed_dict

    def get_values(self) -> List[Value]:
        return self.values

    def get_conditions(self) -> List[Value]:
        return self.conditions

    @tensorflow_name_scoped
    def unified_inputs(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        # Concatenate input tensors per value
        x = tf.concat(values=[
            value.unify_inputs(xs=[inputs[name] for name in value.learned_input_columns()])
            for value in self.values if value.learned_input_size() > 0
        ], axis=1)

        return x

    @tensorflow_name_scoped
    def add_conditions(self, x: tf.Tensor, conditions: Dict[str, tf.Tensor]) -> tf.Tensor:
        if len(self.conditions) > 0:
            # Condition c
            c = tf.concat(values=[
                value.unify_inputs(xs=[conditions[name] for name in value.learned_input_columns()])
                for value in self.conditions
            ], axis=1)

            # Concatenate z,c
            x = tf.concat(values=(x, c), axis=1)

        return x

    @tensorflow_name_scoped
    def value_losses(self, y: tf.Tensor, inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        # Split output tensors per value
        ys = tf.split(
            value=y, num_or_size_splits=[value.learned_output_size() for value in self.values],
            axis=1
        )

        losses: Dict[str, tf.Tensor] = OrderedDict()

        # Reconstruction loss per value
        for value, y in zip(self.values, ys):
            losses[value.name + '-loss'] = value.loss(
                y=y, xs=[inputs[name] for name in value.learned_output_columns()]
            )

        # Regularization loss
        reg_losses = tf.compat.v1.losses.get_regularization_losses()
        if len(reg_losses) > 0:
            losses['regularization-loss'] = tf.add_n(inputs=reg_losses, name='regularization_loss')
        else:
            losses['regularization-loss'] = tf.constant(0, dtype=tf.float32)

        # Reconstruction loss
        losses['reconstruction-loss'] = tf.add_n(inputs=list(losses.values()), name='reconstruction_loss')

        return losses

    def get_column_names(self) -> List[str]:
        columns = [
            name for value in (self.values + self.conditions)
            for name in value.learned_output_columns()
        ]
        return columns

    def _apply_type_overrides(self, df, name) -> Value:
        assert name in self.type_overrides
        forced_type = self.type_overrides[name]
        if forced_type == TypeOverride.ID:
            value: Value = self.create_identifier(name)
            self.identifier_value = value
        elif forced_type == TypeOverride.CATEGORICAL:
            value = self.create_categorical(name)
        elif forced_type == TypeOverride.CONTINUOUS:
            value = self.create_continuous(name)
        elif forced_type == TypeOverride.DATE:
            value = self.create_date(name)
        elif forced_type == TypeOverride.ENUMERATION:
            value = self.create_enumeration(name)
        else:
            assert False
        is_nan = df[name].isna().any()
        if is_nan and isinstance(value, ContinuousValue):
            value = self.create_nan(name, value)
        return value

    def create_identifier(self, name: str) -> IdentifierValue:
        """Create IdentifierValue."""
        return IdentifierValue(name=name, capacity=self.capacity)

    def create_categorical(self, name: str, **kwargs) -> CategoricalValue:
        """Create CategoricalValue."""
        categorical_kwargs = dict(self.categorical_kwargs)
        categorical_kwargs['produce_nans'] = True if name in self.produce_nans_for else False
        categorical_kwargs.update(kwargs)
        return CategoricalValue(name=name, **categorical_kwargs)

    def create_continuous(self, name: str, **kwargs) -> ContinuousValue:
        """Create ContinuousValue."""
        continuous_kwargs = dict(self.continuous_kwargs)
        continuous_kwargs.update(kwargs)
        return ContinuousValue(name=name, **continuous_kwargs)

    def create_date(self, name: str) -> DateValue:
        """Create DateValue."""
        return DateValue(
            name=name, categorical_kwargs=self.categorical_kwargs,
            **self.continuous_kwargs
        )

    def create_nan(self, name: str, value: Value) -> NanValue:
        """Create NanValue."""
        nan_kwargs = dict(self.nan_kwargs)
        nan_kwargs['produce_nans'] = True if name in self.produce_nans_for else False
        return NanValue(name=name, value=value, **nan_kwargs)

    def create_person(self) -> PersonValue:
        """Create PersonValue."""
        return PersonValue(
            name='person', title_label=self.title_label,
            gender_label=self.gender_label,
            name_label=self.name_label, firstname_label=self.firstname_label,
            lastname_label=self.lastname_label, email_label=self.email_label,
            mobile_number_label=self.mobile_number_label,
            home_number_label=self.home_number_label,
            work_number_label=self.work_number_label,
            categorical_kwargs=self.categorical_kwargs
        )

    def create_bank(self) -> BankNumberValue:
        """Create BankNumberValue."""
        return BankNumberValue(
            name='bank',
            bic_label=self.bic_label,
            sort_code_label=self.sort_code_label,
            account_label=self.account_label
        )

    def create_compound_address(self) -> CompoundAddressValue:
        """Create CompoundAddressValue."""
        return CompoundAddressValue(
            name='address', postcode_level=1,
            address_label=self.address_label, postcode_regex=self.postcode_regex,
            capacity=self.capacity
        )

    def create_address(self) -> AddressValue:
        """Create AddressValue."""
        return AddressValue(
            name='address', postcode_level=0,
            postcode_label=self.postcode_label, county_label=self.county_label,
            city_label=self.city_label, district_label=self.district_label,
            street_label=self.street_label, house_number_label=self.house_number_label,
            flat_label=self.flat_label, house_name_label=self.house_name_label,
            categorical_kwargs=self.categorical_kwargs
        )

    def create_enumeration(self, name: str) -> EnumerationValue:
        """Create EnumerationValue."""
        return EnumerationValue(name=name)

    def create_sampling(self, name: str) -> SamplingValue:
        """Create SamplingValue."""
        return SamplingValue(name=name)

    def create_constant(self, name: str) -> ConstantValue:
        """Create ConstantValue."""
        return ConstantValue(name=name)

    def identify_value(self, col: pd.Series, name: str) -> Optional[Value]:
        """Autodetect the type of a column and assign a name.

        Args:
            col: A column from DataFrame.
            name: A name to give to the value.

        Returns: Detected value or None which means that the value has already been detected before.

        """
        value: Optional[Value] = None

        # ========== Pre-configured values ==========

        # Person value
        if name in [self.title_label, self.gender_label, self.name_label, self.firstname_label, self.lastname_label,
                    self.email_label, self.mobile_number_label, self.home_number_label, self.work_number_label]:
            if self.person_value is None:
                value = self.create_person()
                self.person_value = value
            else:
                return None

        # Bank value
        elif name in [self.bic_label, self.sort_code_label, self.account_label]:
            if self.bank_value is None:
                value = self.create_bank()
                self.bank_value = value
            else:
                return None

        # Address value
        elif name in [self.postcode_label, self.county_label, self.city_label, self.district_label, self.street_label,
                      self.house_number_label, self.flat_label, self.house_name_label]:
            if self.address_value is None:
                value = self.create_address()
                self.address_value = value
            else:
                return None

        # Compound address value
        elif name == self.address_label:
            if self.address_value is None:
                value = self.create_compound_address()
                self.address_value = value
            else:
                return None

        # Identifier value
        elif name == self.identifier_label:
            if self.identifier_value is None:
                value = self.create_identifier(name)
                self.identifier_value = value
            else:
                return None

        # Return pre-configured value
        if value is not None:
            return value

        # ========== Non-numeric values ==========

        num_data = len(col)
        num_unique = col.nunique()
        is_nan = False

        if num_unique == 1:
            return self.create_constant(name)

        # Categorical value if small number of distinct values
        elif num_unique <= CATEGORICAL_THRESHOLD_LOG_MULTIPLIER * log(num_data):
            # is_nan = df.isna().any()
            if _column_does_not_contain_genuine_floats(col):
                if num_unique > 2:
                    value = self.create_categorical(name, similarity_based=True)
                else:
                    value = self.create_categorical(name)

        # Date value
        elif col.dtype.kind == 'M':  # 'm' timedelta
            is_nan = col.isna().any()
            value = self.create_date(name)

        # Boolean value
        elif col.dtype.kind == 'b':
            # is_nan = df.isna().any()
            value = self.create_categorical(name, categories=[False, True])

        # Continuous value if integer (reduced variability makes similarity-categorical more likely)
        elif col.dtype.kind == 'i':
            value = self.create_continuous(name, integer=True)

        # Categorical value if object type has attribute 'categories'
        elif col.dtype.kind == 'O' and hasattr(col.dtype, 'categories'):
            # is_nan = df.isna().any()
            if num_unique > 2:
                value = self.create_categorical(name, pandas_category=True, categories=col.dtype.categories,
                                                similarity_based=True)
            else:
                value = self.create_categorical(name, pandas_category=True, categories=col.dtype.categories)

        # Date value if object type can be parsed
        elif col.dtype.kind == 'O':
            try:
                date_data = pd.to_datetime(col)
                num_nan = date_data.isna().sum()
                if num_nan / num_data < PARSING_NAN_FRACTION_THRESHOLD:
                    assert date_data.dtype.kind == 'M'
                    value = self.create_date(name)
                    is_nan = num_nan > 0
            except (ValueError, TypeError, OverflowError):
                pass

        # Similarity-based categorical value if not too many distinct values
        elif num_unique <= sqrt(num_data):
            if _column_does_not_contain_genuine_floats(col):
                if num_unique > 2:
                    value = self.create_categorical(name, similarity_based=True)
                else:
                    value = self.create_categorical(name)

        # Return non-numeric value and handle NaNs if necessary
        if value is not None:
            if is_nan:
                value = self.create_nan(name, value)
            return value

        # ========== Numeric value ==========

        # Try parsing if object type
        if col.dtype.kind == 'O':
            numeric_data = pd.to_numeric(col, errors='coerce')
            num_nan = numeric_data.isna().sum()
            if num_nan / num_data < PARSING_NAN_FRACTION_THRESHOLD:
                assert numeric_data.dtype.kind in ('f', 'i')
                is_nan = num_nan > 0
            else:
                numeric_data = col
                is_nan = col.isna().any()
        elif col.dtype.kind in ('f', 'i'):
            numeric_data = col
            is_nan = col.isna().any()
        # Return numeric value and handle NaNs if necessary
        if numeric_data.dtype.kind in ('f', 'i'):
            value = self.create_continuous(name)
            if is_nan:
                value = self.create_nan(name, value)
            return value

        # ========== Fallback values ==========

        # Enumeration value if strictly increasing
        if col.dtype.kind != 'f' and num_unique == num_data and col.is_monotonic_increasing:
            value = self.create_enumeration(name)

        # Sampling value otherwise
        else:
            value = self.create_sampling(name)

        assert value is not None
        return value


def _column_does_not_contain_genuine_floats(col: pd.Series) -> bool:
    """
    Returns TRUE of the input column contains genuine floats, that would exclude integers with type float.
        e.g.:
            _column_does_not_contain_genuine_floats(['A', 'B', 'C']) returns True
            _column_does_not_contain_genuine_floats([1.0, 3.0, 2.0]) returns True
            _column_does_not_contain_genuine_floats([1.0, 3.2, 2.0]) returns False

    :param col: input pd.Series
    :return: bool
    """

    return not col.dropna().apply(_is_not_integer_float).any()


def _is_not_integer_float(x: float) -> bool:
    """
    Returns whether 'x' is a float and is not integer.
        e.g.:
            _is_not_integer_float(3.0) = False
            _is_not_integer_float(3.2) = True

    :param x: input float
    :return: bool
    """

    if type(x) == float:
        return not x.is_integer()
    else:
        return False
