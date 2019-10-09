"""Utilities that help you create Value objects."""

from math import log, sqrt
from typing import Dict, Any, Optional, cast

import pandas as pd

from .address import AddressValue
from .categorical import CategoricalValue
from .compound_address import CompoundAddressValue
from .continuous import ContinuousValue
from .date import DateValue
from .enumeration import EnumerationValue
from .identifier import IdentifierValue
from .nan import NanValue
from .person import PersonValue
from .sampling import SamplingValue
from .value import Module
from .value import Value

CATEGORICAL_THRESHOLD_LOG_MULTIPLIER = 2.5
PARSING_NAN_FRACTION_THRESHOLD = 0.25


class ValueFactory(Module):
    """A Mix-In that you extend to be able to create various values."""

    def __init__(self):
        """Init ValueFactory."""
        # type hack to allow dynamic access to properties
        self.module = cast(Any, self)

        categorical_kwargs: Dict[str, Any] = dict()
        continuous_kwargs: Dict[str, Any] = dict()
        nan_kwargs: Dict[str, Any] = dict()
        categorical_kwargs['capacity'] = self.module.capacity
        nan_kwargs['capacity'] = self.module.capacity
        categorical_kwargs['weight_decay'] = self.module.weight_decay
        nan_kwargs['weight_decay'] = self.module.weight_decay
        categorical_kwargs['weight'] = self.module.categorical_weight
        nan_kwargs['weight'] = self.module.categorical_weight
        continuous_kwargs['weight'] = self.module.continuous_weight
        categorical_kwargs['temperature'] = self.module.temperature
        categorical_kwargs['smoothing'] = self.module.smoothing
        categorical_kwargs['moving_average'] = self.module.moving_average
        categorical_kwargs['similarity_regularization'] = self.module.similarity_regularization
        categorical_kwargs['entropy_regularization'] = self.module.entropy_regularization

        self.categorical_kwargs = categorical_kwargs
        self.continuous_kwargs = continuous_kwargs
        self.nan_kwargs = nan_kwargs

    def create_identifier(self, name: str) -> IdentifierValue:
        """Create IdentifierValue."""
        return self.add_module(
            module='identifier', name=name, capacity=self.module.capacity
        )

    def create_categorical(self, name: str, **kwargs) -> CategoricalValue:
        """Create CategoricalValue."""
        categorical_kwargs = dict(self.categorical_kwargs)
        categorical_kwargs.update(kwargs)
        return self.add_module(module='categorical', name=name, **categorical_kwargs)

    def create_continuous(self, name: str, **kwargs) -> ContinuousValue:
        """Create ContinuousValue."""
        continuous_kwargs = dict(self.continuous_kwargs)
        continuous_kwargs.update(kwargs)
        return self.add_module(module='continuous', name=name, **continuous_kwargs)

    def create_date(self, name: str) -> DateValue:
        """Create DateValue."""
        return self.add_module(
            module='date', name=name, categorical_kwargs=self.categorical_kwargs,
            **self.continuous_kwargs
        )

    def create_nan(self, name: str, value: Value) -> NanValue:
        """Create NanValue."""
        nan_kwargs = dict(self.nan_kwargs)
        nan_kwargs['produce_nans'] = True if name in self.module.produce_nans_for else False
        return self.add_module(module='nan', name=name, value=value, **nan_kwargs)

    def create_person(self) -> PersonValue:
        """Create PersonValue."""
        return self.add_module(
            module='person', name='person', title_label=self.module.title_label,
            gender_label=self.module.gender_label,
            name_label=self.module.name_label, firstname_label=self.module.firstname_label,
            lastname_label=self.module.lastname_label, email_label=self.module.email_label,
            capacity=self.module.capacity, weight_decay=self.module.weight_decay
        )

    def create_compound_address(self) -> CompoundAddressValue:
        """Create CompoundAddressValue."""
        return self.add_module(
            module='compound_address', name='address', postcode_level=1,
            address_label=self.module.address_label, postcode_regex=self.module.postcode_regex,
            capacity=self.module.capacity, weight_decay=self.module.weight_decay
        )

    def create_address(self) -> AddressValue:
        """Create AddressValue."""
        return self.add_module(
            module='address', name='address', postcode_level=0,
            postcode_label=self.module.postcode_label, city_label=self.module.city_label,
            street_label=self.module.street_label,
            capacity=self.module.capacity, weight_decay=self.module.weight_decay
        )

    def create_enumeration(self, name: str) -> EnumerationValue:
        """Create EnumerationValue."""
        return self.add_module(module='enumeration', name=name)

    def create_sampling(self, name: str) -> SamplingValue:
        """Create SamplingValue."""
        return self.module.add_module(module='sampling', name=name)

    def identify_value(self, col: pd.Series, name: str) -> Value:
        """Autodetect the type of a column and assign a name.

        Args:
            col: A column from DataFrame.
            name: A name to give to the value.

        Returns: Detected value.

        """
        value: Optional[Value] = None

        # ========== Pre-configured values ==========

        # Person value
        if name == getattr(self.module, 'title_label', None) or \
                name == getattr(self.module, 'gender_label', None) or \
                name == getattr(self.module, 'name_label', None) or \
                name == getattr(self.module, 'firstname_label', None) or \
                name == getattr(self.module, 'lastname_label', None) or \
                name == getattr(self.module, 'email_label', None):
            if self.module.person_value is None:
                value = self.create_person()
                self.module.person_value = value

        # Address value
        elif name == getattr(self.module, 'postcode_label', None) or \
                name == getattr(self.module, 'city_label', None) or \
                name == getattr(self.module, 'street_label', None):
            if self.module.address_value is None:
                value = self.create_address()
                self.module.address_value = value

        # Compound address value
        elif name == getattr(self.module, 'address_label', None):
            value = self.create_compound_address()
            self.module.address_value = value

        # Identifier value
        elif name == getattr(self.module, 'identifier_label', None):
            value = self.create_identifier(name)
            self.module.identifier_value = value

        # Return pre-configured value
        if value is not None:
            return value

        # ========== Non-numeric values ==========

        num_data = len(col)
        num_unique = col.nunique()
        is_nan = False

        # Categorical value if small number of distinct values
        if num_unique <= CATEGORICAL_THRESHOLD_LOG_MULTIPLIER * log(num_data):
            # is_nan = df.isna().any()
            if _column_does_not_contain_genuine_floats(col):
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
                value = self.create_categorical(name, similarity_based=True)

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
