"""Utilities that help you create Value objects."""
import enum
from math import log, sqrt
from typing import Dict, Any, Optional, Union, Iterable, List, Set, Tuple
import logging

import numpy as np
import pandas as pd
import treelib as tl

from .address import AddressValue
from .associated_categorical import AssociatedCategoricalValue
from .bank_number import BankNumberValue
from .categorical import CategoricalValue
from .compound_address import CompoundAddressValue
from .constant import ConstantValue
from .continuous import ContinuousValue
from .date import DateValue
from .decomposed_continuous import DecomposedContinuousValue
from .enumeration import EnumerationValue
from .identifier import IdentifierValue
from .identify_rules import identify_rules
from .nan import NanValue
from .person import PersonValue
from .sampling import SamplingValue
from .value import Value

CATEGORICAL_THRESHOLD_LOG_MULTIPLIER = 2.5
PARSING_NAN_FRACTION_THRESHOLD = 0.25


logger = logging.getLogger(__name__)


class TypeOverride(enum.Enum):
    ID = 'ID'
    DATE = 'DATE'
    CATEGORICAL = 'CATEGORICAL'
    CONTINUOUS = 'CONTINUOUS'
    ENUMERATION = 'ENUMERATION'


class ValueFactory:
    """A Mix-In that you extend to be able to create various values."""

    def __init__(
        self, df: pd.DataFrame, capacity: int = 128,
        continuous_weight: float = 1.0, decompose_continuous_values: bool = False,
        categorical_weight: float = 1.0, temperature: float = 1.0, moving_average: bool = True, nan_weight: float = 1.0,
        keep_monotonic_dates: bool = False,
        name: str = 'value_factory',
        type_overrides: Dict[str, TypeOverride] = None,
        produce_nans_for: Union[bool, Iterable[str], None] = None,
        column_aliases: Dict[str, str] = None, condition_columns: List[str] = None,
        find_rules: Union[str, List[str]] = None, associations: Dict[str, List[str]] = None,
        # Person
        title_label: Union[str, List[str]] = None, gender_label: Union[str, List[str]] = None,
        name_label: Union[str, List[str]] = None, firstname_label: Union[str, List[str]] = None,
        lastname_label: Union[str, List[str]] = None, email_label: Union[str, List[str]] = None,
        mobile_number_label: Union[str, List[str]] = None, home_number_label: Union[str, List[str]] = None,
        work_number_label: Union[str, List[str]] = None,
        # Bank
        bic_label: Union[str, List[str]] = None, sort_code_label: Union[str, List[str]] = None,
        account_label: Union[str, List[str]] = None,
        # Address
        postcode_label: Union[str, List[str]] = None, county_label: Union[str, List[str]] = None,
        city_label: Union[str, List[str]] = None, district_label: Union[str, List[str]] = None,
        street_label: Union[str, List[str]] = None, house_number_label: Union[str, List[str]] = None,
        flat_label: Union[str, List[str]] = None, house_name_label: Union[str, List[str]] = None,
        addresses_file: str = None,
        # Compound Address
        address_label: str = None, postcode_regex: str = None,
        # Identifier
        identifier_label: str = None,
    ):

        """Init ValueFactory."""
        self.name = name

        categorical_kwargs: Dict[str, Any] = dict()
        continuous_kwargs: Dict[str, Any] = dict()
        nan_kwargs: Dict[str, Any] = dict()
        date_kwargs: Dict[str, Any] = dict()

        continuous_kwargs['weight'] = continuous_weight
        categorical_kwargs['capacity'] = capacity
        categorical_kwargs['weight'] = categorical_weight
        categorical_kwargs['temperature'] = temperature
        categorical_kwargs['moving_average'] = moving_average
        nan_kwargs['capacity'] = capacity
        nan_kwargs['weight'] = nan_weight
        date_kwargs['keep_monotonic'] = keep_monotonic_dates

        self.categorical_kwargs = categorical_kwargs
        self.continuous_kwargs = continuous_kwargs
        self.nan_kwargs = nan_kwargs
        self.date_kwargs = date_kwargs

        self.decompose_continuous_values = decompose_continuous_values

        if find_rules is None:
            self.find_rules: Union[str, List[str]] = []
        else:
            self.find_rules = find_rules

        # Values
        self.columns = list(df.columns)
        self.values: List[Value] = list()
        self.conditions: List[Value] = list()

        self.capacity = capacity

        # Person
        self.title_label = _get_formated_label(title_label)
        self.gender_label = _get_formated_label(gender_label)
        self.name_label = _get_formated_label(name_label)
        self.firstname_label = _get_formated_label(firstname_label)
        self.lastname_label = _get_formated_label(lastname_label)
        self.email_label = _get_formated_label(email_label)
        self.mobile_number_label = _get_formated_label(mobile_number_label)
        self.home_number_label = _get_formated_label(home_number_label)
        self.work_number_label = _get_formated_label(work_number_label)

        person_labels = [
            self.title_label, self.gender_label, self.name_label, self.firstname_label, self.lastname_label,
            self.email_label, self.mobile_number_label, self.home_number_label, self.work_number_label
        ]
        self.person_labels = _get_labels_matrix(person_labels)
        self.person_values: List[Optional[Value]] = [None] * len(self.person_labels)

        # Bank Number
        self.bic_label = _get_formated_label(bic_label)
        self.sort_code_label = _get_formated_label(sort_code_label)
        self.account_label = _get_formated_label(account_label)

        bank_labels = [self.bic_label, self.sort_code_label, self.account_label]
        self.bank_labels = _get_labels_matrix(bank_labels)
        self.bank_values: List[Optional[Value]] = [None] * len(self.bank_labels)

        # Address
        self.postcode_label = _get_formated_label(postcode_label)
        self.county_label = _get_formated_label(county_label)
        self.city_label = _get_formated_label(city_label)
        self.district_label = _get_formated_label(district_label)
        self.street_label = _get_formated_label(street_label)
        self.house_number_label = _get_formated_label(house_number_label)
        self.flat_label = _get_formated_label(flat_label)
        self.house_name_label = _get_formated_label(house_name_label)
        self.addresses_file = addresses_file

        address_labels = [
            self.postcode_label, self.county_label, self.city_label, self.district_label, self.street_label,
            self.house_number_label, self.flat_label, self.house_name_label
        ]
        self.address_labels = _get_labels_matrix(address_labels)
        self.address_values: List[Optional[Value]] = [None] * len(self.address_labels)

        # Compound Address
        self.address_value: Optional[Value] = None
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

        associated_values = []
        association_tree = tl.Tree()

        if associations is not None:
            association_tree.create_node('root', 'root')
            for n, nodes in associations.items():
                if association_tree.get_node(n) is None:
                    association_tree.create_node(n, n, 'root')
                for m in nodes:
                    if association_tree.get_node(m) is None:
                        association_tree.create_node(m, m, n)
                    else:
                        association_tree.move_node(m, n)

        associates = [n for n in association_tree.expand_tree('root')][1:]
        association_groups = [st[1:] for st in association_tree.paths_to_leaves()]

        for name in df.columns:
            # we are skipping aliases
            if name in self.column_aliases:
                continue
            if name in self.type_overrides:
                value = self._apply_type_overrides(df, name)
            else:
                try:
                    identified_value, reason = self.identify_value(col=df[name], name=name)
                    # None means the value has already been detected:
                    if identified_value is None:
                        continue

                    logger.debug("Identified column %s (%s:%s) as %s. Reason: %s", name, df[name].dtype,
                                 df[name].dtype.kind, identified_value.__class__.__name__, reason)
                except Exception as e:
                    logger.error("Failed to identify column %s (%s:%s).", name, df[name].dtype,
                                 df[name].dtype.kind)
                    raise e

                value = identified_value
            if name in self.condition_columns:
                self.conditions.append(value)
            elif name == self.identifier_label:
                self.identifier_value = value
            elif name in associates:
                associated_values.append(value)
            else:
                self.values.append(value)

        if len(associated_values) > 0:
            self.values.append(AssociatedCategoricalValue(values=associated_values, associations=association_groups))

        # Automatic extraction of specification parameters
        df = df.copy()
        for value in self.all_values:
            value.extract(df=df)

        # Identify deterministic rules
        #  import ipdb; ipdb.set_trace()
        self.values = identify_rules(values=self.values, df=df, tests=self.find_rules)

    @property
    def all_values(self):
        if self.identifier_value:
            return self.values + self.conditions + [self.identifier_value]
        else:
            return self.values + self.conditions

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a preprocessed copy of the input DataFrame"""
        df_copy = df.copy()
        for value in self.all_values:
            df_copy = value.preprocess(df=df_copy)

        return df_copy

    def postprocess(self,  df: pd.DataFrame) -> pd.DataFrame:
        """Post-processes the input DataFrame"""
        for value in self.all_values:
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

    def create_continuous(self, name: str, **kwargs) -> Union[ContinuousValue, DecomposedContinuousValue]:
        """Create ContinuousValue."""
        continuous_kwargs = dict(self.continuous_kwargs)
        continuous_kwargs.update(kwargs)
        if self.decompose_continuous_values:
            return DecomposedContinuousValue(name=name, identifier=self.identifier_label, **continuous_kwargs)
        else:
            return ContinuousValue(name=name, **continuous_kwargs)

    def create_date(self, name: str) -> DateValue:
        """Create DateValue."""
        return DateValue(
            name=name, categorical_kwargs=self.categorical_kwargs, continuous_kwargs=self.continuous_kwargs,
            **self.date_kwargs
        )

    def create_nan(self, name: str, value: Value) -> NanValue:
        """Create NanValue."""
        nan_kwargs = dict(self.nan_kwargs)
        nan_kwargs['produce_nans'] = True if name in self.produce_nans_for else False
        return NanValue(name=name, value=value, **nan_kwargs)

    def create_person(self, i: int) -> PersonValue:
        """Create PersonValue."""
        return PersonValue(
            name='person_{}'.format(i),
            title_label=self.title_label[i] if self.title_label else None,
            gender_label=self.gender_label[i] if self.gender_label else None,
            name_label=self.name_label[i] if self.name_label else None,
            firstname_label=self.firstname_label[i] if self.firstname_label else None,
            lastname_label=self.lastname_label[i] if self.lastname_label else None,
            email_label=self.email_label[i] if self.email_label else None,
            mobile_number_label=self.mobile_number_label[i] if self.mobile_number_label else None,
            home_number_label=self.home_number_label[i] if self.home_number_label else None,
            work_number_label=self.work_number_label[i] if self.work_number_label else None,
            categorical_kwargs=self.categorical_kwargs
        )

    def create_bank(self, i: int) -> BankNumberValue:
        """Create BankNumberValue."""
        return BankNumberValue(
            name='bank_{}'.format(i),
            bic_label=self.bic_label[i] if self.bic_label else None,
            sort_code_label=self.sort_code_label[i] if self.sort_code_label else None,
            account_label=self.account_label[i] if self.account_label else None
        )

    def create_compound_address(self) -> CompoundAddressValue:
        """Create CompoundAddressValue."""
        return CompoundAddressValue(
            name='address', postcode_level=1,
            address_label=self.address_label, postcode_regex=self.postcode_regex,
            capacity=self.capacity
        )

    def create_address(self, i: int) -> AddressValue:
        """Create AddressValue."""
        return AddressValue(
            name='address_{}'.format(i), postcode_level=0,
            postcode_label=self.postcode_label[i] if self.postcode_label else None,
            county_label=self.county_label[i] if self.county_label else None,
            city_label=self.city_label[i] if self.city_label else None,
            district_label=self.district_label[i] if self.district_label else None,
            street_label=self.street_label[i] if self.street_label else None,
            house_number_label=self.house_number_label[i] if self.house_number_label else None,
            flat_label=self.flat_label[i] if self.flat_label else None,
            house_name_label=self.house_name_label[i] if self.house_name_label else None,
            addresses_file=self.addresses_file, categorical_kwargs=self.categorical_kwargs
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

    def identify_value(self, col: pd.Series, name: str) -> Tuple[Optional[Value], Optional[str]]:
        """Autodetect the type of a column and assign a name.

        Args:
            col: A column from DataFrame.
            name: A name to give to the value.

        Returns: Detected value or None which means that the value has already been detected before.

        """
        if str(col.dtype) == 'category':
            col = col.astype(object).infer_objects()

        value: Optional[Value] = None
        reason: str = ""

        # ========== Pre-configured values ==========

        # Person value
        if len(self.person_labels) > 0 and name in np.concatenate(self.person_labels):
            for i, person in enumerate(self.person_labels):
                if name in person:
                    if self.person_values[i] is None:
                        value = self.create_person(i)
                        self.person_values[i] = value
                    else:
                        return None, None

        # Bank value
        elif len(self.bank_labels) > 0 and name in np.concatenate(self.bank_labels):
            for i, bank in enumerate(self.bank_labels):
                if name in bank:
                    if self.bank_values[i] is None:
                        value = self.create_bank(i)
                        self.bank_values[i] = value
                    else:
                        return None, None

        # Address value
        elif len(self.address_labels) > 0 and name in np.concatenate(self.address_labels):
            for i, address in enumerate(self.address_labels):
                if name in address:
                    if self.address_values[i] is None:
                        value = self.create_address(i)
                        self.address_values[i] = value
                    else:
                        return None, None

        # Compound address value
        elif name == self.address_label:
            if self.address_value is None:
                value = self.create_compound_address()
                self.address_value = value
            else:
                return None, None

        # Identifier value
        elif name == self.identifier_label:
            if self.identifier_value is None:
                value = self.create_identifier(name)
                self.identifier_value = value
            else:
                return None, None

        # Return pre-configured value
        if value is not None:
            return value, "Name matched preconfigured label. "

        # ========== Non-numeric values ==========

        num_data = len(col)
        num_unique = col.nunique(dropna=False)
        is_nan = False

        excl_nan_dtype = col[col.notna()].infer_objects().dtype

        if num_unique <= 1:
            return self.create_constant(name), "num_unique <= 1. "

        # Categorical value if small number of distinct values
        elif num_unique <= CATEGORICAL_THRESHOLD_LOG_MULTIPLIER * log(num_data):
            # is_nan = df.isna().any()
            if _column_does_not_contain_genuine_floats(col):
                if num_unique > 2:
                    value = self.create_categorical(name, similarity_based=True, true_categorical=True)
                    reason = "Small (< log(N)) number of distinct values. "
                else:
                    value = self.create_categorical(name, true_categorical=True)
                    reason = "Small (< log(N)) number of distinct values (= 2). "

        # Date value
        elif col.dtype.kind == 'M':  # 'm' timedelta
            is_nan = col.isna().any()
            value = self.create_date(name)
            reason = "Column dtype kind is 'M'. "

        # Boolean value
        elif col.dtype.kind == 'b':
            # is_nan = df.isna().any()
            value = self.create_categorical(name, categories=[False, True], true_categorical=True)
            reason = "Column dtype kind is 'b'. "

        # Continuous value if integer (reduced variability makes similarity-categorical more likely)
        elif col.dtype.kind in ['i', 'u']:
            value = self.create_continuous(name, integer=True)
            reason = f"Column dtype kind is '{col.dtype.kind}'. "

        # Categorical value if object type has attribute 'categories'
        elif col.dtype.kind == 'O' and hasattr(col.dtype, 'categories'):
            # is_nan = df.isna().any()
            if num_unique > 2:
                value = self.create_categorical(name, pandas_category=True, similarity_based=True,
                                                true_categorical=True)
                reason = "Column dtype kind is 'O' and has 'categories' (> 2). "
            else:
                value = self.create_categorical(name, pandas_category=True, true_categorical=True)
                reason = "Column dtype kind is 'O' and has 'categories' (= 2). "

        # Date value if object type can be parsed
        elif col.dtype.kind == 'O' and excl_nan_dtype.kind not in ['f', 'i']:
            try:
                date_data = pd.to_datetime(col)
                num_nan = date_data.isna().sum()
                if num_nan / num_data < PARSING_NAN_FRACTION_THRESHOLD:
                    assert date_data.dtype.kind == 'M'
                    value = self.create_date(name)
                    reason = "Column dtype is 'O' and convertable to datetime. "
                    is_nan = num_nan > 0
            except (ValueError, TypeError, OverflowError):
                pass

        # Similarity-based categorical value if not too many distinct values
        elif num_unique <= sqrt(num_data):  # num_data must be > 161 to be true.
            if _column_does_not_contain_genuine_floats(col):
                if num_unique > 2:  # note the alternative is never possible anyway.
                    value = self.create_categorical(name, similarity_based=True, true_categorical=False)
                    reason = "Small (< sqrt(N)) number of distinct values. "

        # Return non-numeric value and handle NaNs if necessary
        if value is not None:
            if is_nan:
                value = self.create_nan(name, value)
                reason += "And contains NaNs. "
            return value, reason

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
        else:
            numeric_data = None
        # Return numeric value and handle NaNs if necessary
        if numeric_data is not None and numeric_data.dtype.kind in ('f', 'i'):
            value = self.create_continuous(name)
            reason = f"Converted to numeric dtype ({numeric_data.dtype.kind}) with success " + \
                     f"rate > {1.0-PARSING_NAN_FRACTION_THRESHOLD}. "
            if is_nan:
                value = self.create_nan(name, value)
                reason += " And contains NaNs. "
            return value, reason

        # ========== Fallback values ==========

        # Sampling value otherwise
        value = self.create_sampling(name)
        reason = "No other criteria met. "

        assert value is not None
        return value, reason


def _column_does_not_contain_genuine_floats(col: pd.Series) -> bool:
    """Returns TRUE of the input column contains genuine floats, that would exclude integers with type float.

        e.g.:
            _column_does_not_contain_genuine_floats(['A', 'B', 'C']) returns True
            _column_does_not_contain_genuine_floats([1.0, 3.0, 2.0]) returns True
            _column_does_not_contain_genuine_floats([1.0, 3.2, 2.0]) returns False

    :param col: input pd.Series
    :return: bool
    """

    return not col.dropna().apply(_is_not_integer_float).any()


def _is_not_integer_float(x) -> bool:
    """Returns whether 'x' is a float and is not integer.

        e.g.:
            _is_not_integer_float(3.0) = False
            _is_not_integer_float(3.2) = True

    :param x: input
    :return: bool
    """

    if type(x) == float:
        return not x.is_integer()
    else:
        return False


def _get_formated_label(label: Union[str, List[str], None]) -> Union[List[str], None]:
    """Change the format of a label, if its string return [string], otherwise return itself."""
    if isinstance(label, str):
        return [label]
    else:
        return label


def _get_labels_matrix(labels: List[Optional[List[str]]]) -> np.array:
    """From a list of labels, check if the sizes are consistent and return the matrix of labels,
    with shape (num_values, num_labels).

        e.g.: If we have 2 addresses with 5 labels the output shape will be (2, 5).

    """

    labels_len = None
    out_labels = []

    for label in labels:
        if label:
            if labels_len:
                assert labels_len == len(label), 'All labels must have the same lenght'
            else:
                labels_len = len(label)
            out_labels.append(label)

    if len(out_labels) > 0:
        return np.transpose(out_labels)
    else:
        return np.array([])
