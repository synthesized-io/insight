import enum
import logging
from math import sqrt, log
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import treelib as tl
from dataclasses import fields

from .address import AddressMeta
from .association import AssociationMeta
from .bank import BankNumberMeta
from .categorical import CategoricalMeta
from .compound_address import CompoundAddressMeta
from .constant import ConstantMeta
from .continuous import ContinuousMeta
from .data_frame import DataFrameMeta
from .date import DateMeta
from .enumeration import EnumerationMeta
from .identifier import IdentifierMeta
from .identify_rules import identify_rules
from .nan import NanMeta
from .person import PersonMeta
from .sampling import SamplingMeta
from .value_meta import ValueMeta
from ..config import MetaExtractorConfig, AddressParams, BankParams, CompoundAddressParams, PersonParams

logger = logging.getLogger(__name__)


class TypeOverride(enum.Enum):
    ID = 'ID'
    DATE = 'DATE'
    CATEGORICAL = 'CATEGORICAL'
    CONTINUOUS = 'CONTINUOUS'
    ENUMERATION = 'ENUMERATION'


class MetaExtractor:
    def __init__(self, config: MetaExtractorConfig = MetaExtractorConfig()):
        self.config = config

    @classmethod
    def extract(
            cls, df: pd.DataFrame, config: MetaExtractorConfig = MetaExtractorConfig(),
            id_index: str = None, time_index: str = None,
            column_aliases: Dict[str, str] = None, associations: Dict[str, List[str]] = None,
            type_overrides: Dict[str, TypeOverride] = None,
            find_rules: Union[str, List[str]] = None, produce_nans_for: List[str] = None,
            address_params: AddressParams = None, bank_params: BankParams = None,
            compound_address_params: CompoundAddressParams = None,
            person_params: PersonParams = None
    ) -> DataFrameMeta:
        extractor = cls(config)
        dataframe_meta = extractor.extract_dataframe_meta(
            df=df, id_index=id_index, time_index=time_index, column_aliases=column_aliases, associations=associations,
            type_overrides=type_overrides, find_rules=find_rules, produce_nans_for=produce_nans_for,
            address_params=address_params, bank_params=bank_params, compound_address_params=compound_address_params,
            person_params=person_params
        )
        return dataframe_meta

    def extract_dataframe_meta(
            self, df: pd.DataFrame, id_index: str = None, time_index: str = None,
            column_aliases: Dict[str, str] = None, associations: Dict[str, List[str]] = None,
            type_overrides: Dict[str, TypeOverride] = None,
            find_rules: Union[str, List[str]] = None, produce_nans_for: List[str] = None,
            address_params: AddressParams = None, bank_params: BankParams = None,
            compound_address_params: CompoundAddressParams = None,
            person_params: PersonParams = None
    ) -> DataFrameMeta:
        column_aliases = column_aliases or dict()
        associations = associations or dict()
        type_overrides = type_overrides or dict()
        find_rules = find_rules or list()
        produce_nans_for = produce_nans_for or list()

        values: List[ValueMeta] = list()
        identifier_value: Optional[ValueMeta] = None
        time_value: Optional[ValueMeta] = None

        df = df.copy()

        if id_index is not None:
            identifier_value = IdentifierMeta(id_index)
            logger.debug("Adding column %s (%s:%s) as %s for id_value.", id_index, df[id_index].dtype,
                         df[id_index].dtype.kind, identifier_value.__class__.__name__)
            identifier_value.extract(df)
            df = df.drop(id_index, axis=1)
        if time_index is not None:
            time_value = DateMeta(time_index)
            time_value.extract(df)
            df = df.drop(time_index, axis=1)

        if person_params is not None:
            values.extend(self._identify_annotations(df, 'person', person_params, self.config.person_meta_config))
        if bank_params is not None:
            values.extend(self._identify_annotations(df, 'bank', bank_params))
        if address_params is not None:
            values.extend(self._identify_annotations(df, 'address', address_params, self.config.address_meta_config))
        if compound_address_params is not None:
            values.extend(self._identify_annotations(df, 'compound_address', compound_address_params))

        values.extend(self._identify_values(df, column_aliases, type_overrides, find_rules, produce_nans_for))

        association_meta = self.create_associations(values, associations)

        return DataFrameMeta(values=values, id_value=identifier_value, time_value=time_value,
                             column_aliases=column_aliases, association_meta=association_meta)

    @staticmethod
    def _identify_annotations(df: pd.DataFrame, annotation: str, params, config=None):
        labels = {f.name: _get_formated_label(params.__getattribute__(f.name)) for f in fields(params)}

        labels_matrix = _get_labels_matrix([label for label in labels.values()])
        values: List[Optional[ValueMeta]] = list()

        string_to_meta = {
            'bank': BankNumberMeta, 'address': AddressMeta, 'person': PersonMeta,
            'compound_address': CompoundAddressMeta
        }

        if len(labels_matrix) > 0:
            for i, bank in enumerate(labels_matrix):
                kwargs = {k: v[i] if v is not None else None for k, v in labels.items()}
                if config is not None:
                    kwargs['config'] = config
                value = string_to_meta[annotation](
                    name=f'{annotation}_{i}', **kwargs
                )
                values.append(value)

            for value in values:
                value.extract(df=df)

            df.drop(labels=[label for label in np.concatenate(labels_matrix) if label], axis=1, inplace=True)

        return values

    def _identify_values(self, df: pd.DataFrame, column_aliases: Dict[str, str],
                         type_overrides: Dict[str, TypeOverride], find_rules: Union[str, List[str]],
                         produce_nans_for: List[str]):

        values: List[ValueMeta] = list()

        for name in df.columns:
            value: Optional[ValueMeta]
            # we are skipping aliases
            if name in column_aliases:
                logger.debug("Skipping aliased column %s.", name)
                continue
            if name in type_overrides:
                forced_type = type_overrides[name]
                logger.debug("Type Overriding column %s to %s.", name, forced_type)
                if forced_type == TypeOverride.CATEGORICAL:
                    value = CategoricalMeta(name, produce_nans=name in produce_nans_for)
                elif forced_type == TypeOverride.CONTINUOUS:
                    value = ContinuousMeta(name)
                    if any(pd.to_numeric(df[name]).isna()):
                        value = NanMeta(name, value, produce_nans=name in produce_nans_for)
                elif forced_type == TypeOverride.DATE:
                    value = DateMeta(name)
                elif forced_type == TypeOverride.ENUMERATION:
                    value = EnumerationMeta(name)
                else:
                    assert False
            else:
                try:
                    identified_value, reason = self.identify_value(
                        col=df[name], name=name, produce_nans=name in produce_nans_for
                    )
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

            values.append(value)

        # Automatic extraction of specification parameters
        df = df.copy()

        for value in values:
            value.extract(df=df)

        # Identify deterministic rules
        values = identify_rules(values=values, df=df, tests=find_rules)
        return values

    @staticmethod
    def create_associations(value_metas, associations):

        associates = []
        association_groups = []
        association_tree = tl.Tree()
        association_meta: Optional[AssociationMeta] = None

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
            logger.debug(f"Created association tree: {association_tree}")

            associates.extend([n for n in association_tree.expand_tree('root')][1:])
            association_groups.extend([st[1:] for st in association_tree.paths_to_leaves()])

        associated_values = []

        for meta in value_metas:

            if meta.name in associates:
                if isinstance(meta, CategoricalMeta):
                    associated_values.append(meta)
                else:
                    raise ValueError(f"Associated value ({meta.name}) is not a categorical value.")

        if len(associated_values) > 0:
            association_meta = AssociationMeta(values=associated_values, associations=association_groups)

        return association_meta

    def identify_value(self, col: pd.Series, name: str, produce_nans: bool
                       ) -> Tuple[Optional[ValueMeta], Optional[str]]:
        """Autodetect the type of a column and assign a name.

        Returns: Detected value or None which means that the value has already been detected before.

        """
        if str(col.dtype) == 'category':
            col = col.astype(object).infer_objects()

        value: Optional[ValueMeta] = None
        reason: str = ""

        # ========== Non-numeric values ==========

        num_data = len(col)
        num_unique = col.nunique(dropna=False)

        excl_nan_dtype = col[col.notna()].infer_objects().dtype

        if num_unique <= 1:
            return ConstantMeta(name), "num_unique <= 1. "

        # Categorical value if small number of distinct values
        elif num_unique <= self.config.categorical_threshold_log_multiplier * log(num_data):
            # is_nan = df.isna().any()
            if _column_does_not_contain_genuine_floats(col):
                if num_unique > 2:
                    value = CategoricalMeta(
                        name, similarity_based=True, true_categorical=True, produce_nans=produce_nans
                    )
                    reason = "Small (< log(N)) number of distinct values. "
                else:
                    value = CategoricalMeta(name, true_categorical=True, produce_nans=produce_nans)
                    reason = "Small (< log(N)) number of distinct values (= 2). "

        # Date value
        elif col.dtype.kind == 'M':  # 'm' timedelta
            value = DateMeta(name)
            reason = "Column dtype kind is 'M'. "

        # Boolean value
        elif col.dtype.kind == 'b':
            # is_nan = df.isna().any()
            value = CategoricalMeta(
                name, categories=[False, True], true_categorical=True, produce_nans=produce_nans
            )
            reason = "Column dtype kind is 'b'. "

        # Continuous value if integer (reduced variability makes similarity-categorical more likely)
        elif col.dtype.kind in ['i', 'u']:
            value = ContinuousMeta(name, integer=True)
            reason = f"Column dtype kind is '{col.dtype.kind}'. "

        # Categorical value if object type has attribute 'categories'
        elif col.dtype.kind == 'O' and hasattr(col.dtype, 'categories'):
            # is_nan = df.isna().any()
            if num_unique > 2:
                value = CategoricalMeta(name, pandas_category=True, similarity_based=True,
                                        true_categorical=True, produce_nans=produce_nans)
                reason = "Column dtype kind is 'O' and has 'categories' (> 2). "
            else:
                value = CategoricalMeta(name, pandas_category=True, true_categorical=True,
                                        produce_nans=produce_nans)
                reason = "Column dtype kind is 'O' and has 'categories' (= 2). "

        # Date value if object type can be parsed
        elif col.dtype.kind == 'O' and excl_nan_dtype.kind not in ['f', 'i']:
            try:
                date_data = pd.to_datetime(col)
                num_nan = date_data.isna().sum()
                if num_nan / num_data < self.config.parsing_nan_fraction_threshold:
                    assert date_data.dtype.kind == 'M'
                    value = DateMeta(name)
                    reason = "Column dtype is 'O' and convertable to datetime. "
            except (ValueError, TypeError, OverflowError):
                pass

        # Similarity-based categorical value if not too many distinct values
        if value is None and num_unique <= sqrt(num_data):  # num_data must be > 161 to be true.
            if _column_does_not_contain_genuine_floats(col):
                if num_unique > 2:  # note the alternative is never possible anyway.
                    value = CategoricalMeta(name, similarity_based=True, true_categorical=False)
                    reason = "Small (< sqrt(N)) number of distinct values. "

        # Return non-numeric value and handle NaNs if necessary
        if value is not None:
            return value, reason

        # ========== Numeric value ==========
        is_nan: bool = False
        # Try parsing if object type
        if col.dtype.kind == 'O':
            numeric_data = pd.to_numeric(col, errors='coerce')
            num_nan = numeric_data.isna().sum()
            if num_nan / num_data < self.config.parsing_nan_fraction_threshold:
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
            value = ContinuousMeta(name)
            reason = f"Converted to numeric dtype ({numeric_data.dtype.kind}) with success " + \
                     f"rate > {1.0 - self.config.parsing_nan_fraction_threshold}. "
            if is_nan:
                value = NanMeta(name, value, produce_nans)
                reason += " And contains NaNs. "
            return value, reason

        # ========== Fallback values ==========

        # Sampling value otherwise
        value = SamplingMeta(name)
        reason = "No other criteria met. "

        return value, reason


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
                print(labels_len, label)
                assert labels_len == len(label), 'All labels must have the same lenght'
            else:
                labels_len = len(label)
            out_labels.append(label)

    if len(out_labels) > 0:
        return np.transpose(out_labels)
    else:
        return np.array([])


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
