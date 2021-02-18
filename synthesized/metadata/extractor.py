import enum
import logging
from dataclasses import fields
from math import log, sqrt
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import treelib as tl

from .data_frame import DataFrameMeta
from .identify_rules import identify_rules
from .values import (AddressMeta, AssociationMeta, BankNumberMeta, CategoricalMeta, ConstantMeta, ContinuousMeta,
                     DateMeta, EnumerationMeta, FormattedStringMeta, IdentifierMeta, NanMeta, PersonMeta, SamplingMeta,
                     TimeIndexMeta, ValueMeta)
from ..config import AddressParams, BankLabels, FormattedStringParams, MetaExtractorConfig, PersonLabels

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
            find_rules: Union[str, List[str]] = None,
            address_params: AddressParams = None, bank_params: BankLabels = None,
            person_params: PersonLabels = None, formatted_string_params: FormattedStringParams = None,
    ) -> DataFrameMeta:
        extractor = cls(config)
        dataframe_meta = extractor.extract_dataframe_meta(
            df=df, id_index=id_index, time_index=time_index, column_aliases=column_aliases, associations=associations,
            type_overrides=type_overrides, find_rules=find_rules,
            address_params=address_params, bank_params=bank_params, person_params=person_params,
            formatted_string_params=formatted_string_params
        )
        return dataframe_meta

    def extract_dataframe_meta(
            self, df: pd.DataFrame, id_index: str = None, time_index: str = None,
            column_aliases: Dict[str, str] = None, associations: Dict[str, List[str]] = None,
            type_overrides: Dict[str, TypeOverride] = None,
            find_rules: Union[str, List[str]] = None,
            address_params: AddressParams = None, bank_params: BankLabels = None,
            person_params: PersonLabels = None, formatted_string_params: FormattedStringParams = None
    ) -> DataFrameMeta:
        column_aliases = column_aliases or dict()
        associations = associations or dict()
        type_overrides = type_overrides or dict()
        find_rules = find_rules or list()

        values: List[ValueMeta] = list()
        identifier_value: Optional[IdentifierMeta] = None
        time_value: Optional[TimeIndexMeta] = None

        df = df.copy()

        if id_index is not None:
            identifier_value = IdentifierMeta(id_index)
            logger.debug("Adding column %s (%s:%s) as %s for id_value.", id_index, df[id_index].dtype,
                         df[id_index].dtype.kind, identifier_value.__class__.__name__)
            identifier_value.extract(df)
            identifier_value.set_index(df)
        if time_index is not None:
            time_value = TimeIndexMeta(time_index)
            time_value.extract(df)
            time_value.set_index(df)

        values.extend(self._identify_annotations(df, address_params, bank_params,
                                                 person_params, formatted_string_params))

        values.extend(self._identify_values(df, column_aliases, type_overrides, find_rules))

        association_meta = self.create_associations(values, associations)
        if association_meta is not None:
            association_meta.extract(df)

        return DataFrameMeta(values=values, id_value=identifier_value, time_value=time_value,
                             column_aliases=column_aliases, association_meta=association_meta)

    def _identify_annotations(self, df: pd.DataFrame, address_params: AddressParams = None,
                              bank_params: BankLabels = None, person_params: PersonLabels = None,
                              formatted_string_params: FormattedStringParams = None) -> List[ValueMeta]:

        values: List[ValueMeta] = []
        if person_params is not None:
            values.extend(self._identify_annotation(df, meta=PersonMeta, params=person_params, config=self.config.person_meta_config))
        if bank_params is not None:
            values.extend(self._identify_annotation(df, meta=BankNumberMeta, params=bank_params))
        if address_params is not None:
            values.extend(self._identify_annotation(df, meta=AddressMeta, params=address_params, config=self.config.address_meta_config))
        if formatted_string_params is not None:
            values.extend(self._identify_annotation(df, meta=FormattedStringMeta, params=formatted_string_params,
                                                    config=self.config.formatted_string_meta_config))

        return values

    @staticmethod
    def _identify_annotation(df: pd.DataFrame, meta: Type[ValueMeta], params, config=None) -> List[ValueMeta]:
        labels = {f.name: _get_formated_label(params.__getattribute__(f.name)) for f in fields(params)}

        labels_matrix = _get_labels_matrix([label for label in labels.values()])
        values: List[ValueMeta] = list()

        if len(labels_matrix) > 0:
            for i in range(len(labels_matrix)):
                kwargs = {k: v[i] if v is not None else None for k, v in labels.items()}
                if config is not None:
                    kwargs['config'] = config
                value = meta(name=f'{meta.__name__}_{i}', **kwargs)
                values.append(value)

            for value in values:
                value.extract(df=df)

            df.drop(labels=[label for label in np.concatenate(labels_matrix) if label], axis=1, inplace=True)

        return values

    def _identify_values(self, df: pd.DataFrame, column_aliases: Dict[str, str],
                         type_overrides: Dict[str, TypeOverride], find_rules: Union[str, List[str]]):

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
                    value = CategoricalMeta(name)
                elif forced_type == TypeOverride.CONTINUOUS:
                    value = ContinuousMeta(name)
                    if any(pd.to_numeric(df[name]).isin([np.Inf, -np.Inf])) or any(pd.to_numeric(df[name]).isna()):
                        value = NanMeta(name, value)
                elif forced_type == TypeOverride.DATE:
                    value = DateMeta(name)
                elif forced_type == TypeOverride.ENUMERATION:
                    value = EnumerationMeta(name)
                else:
                    assert False
            else:
                try:
                    identified_value, reason = self.identify_value(
                        col=df[name], name=name
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

    def identify_value(self, col: pd.Series, name: str) -> Tuple[Optional[ValueMeta], Optional[str]]:
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

        excl_nan_dtype = col[~col.isin([np.NaN, pd.NaT])].infer_objects().dtype

        if num_unique <= 1:
            return ConstantMeta(name), "num_unique <= 1. "

        # Categorical value if small number of distinct values (or if data-set is too small)
        elif num_unique <= max(float(self.config.min_num_unique),
                               self.config.categorical_threshold_log_multiplier * log(num_data)):
            # is_nan = df.isna().any()
            if _column_does_not_contain_genuine_floats(col):
                if num_unique > 2:
                    value = CategoricalMeta(name, similarity_based=True, true_categorical=True)
                    reason = "Small (< log(N)) number of distinct values. "
                else:
                    value = CategoricalMeta(name, true_categorical=True)
                    reason = "Small (< log(N)) number of distinct values (= 2). "

        # Date value
        elif col.dtype.kind == 'M':  # 'm' timedelta
            value = DateMeta(name)
            reason = "Column dtype kind is 'M'. "

        # Boolean value
        elif col.dtype.kind == 'b':
            # is_nan = df.isna().any()
            value = CategoricalMeta(name, categories=[False, True], true_categorical=True)
            reason = "Column dtype kind is 'b'. "

        # Continuous value if integer (reduced variability makes similarity-categorical more likely)
        elif col.dtype.kind in ['i', 'u']:
            value = ContinuousMeta(name, integer=True)
            reason = f"Column dtype kind is '{col.dtype.kind}'. "

        # Categorical value if object type has attribute 'categories'
        elif col.dtype.kind == 'O' and hasattr(col.dtype, 'categories'):
            # is_nan = df.isna().any()
            if num_unique > 2:
                value = CategoricalMeta(name, pandas_category=True, similarity_based=True, true_categorical=True)
                reason = "Column dtype kind is 'O' and has 'categories' (> 2). "
            else:
                value = CategoricalMeta(name, pandas_category=True, true_categorical=True)
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

                    # deal with NaT (do we need to deal with inf)?
                    if num_nan:
                        value = NanMeta(name, value)
                        reason += " And contains NaTs. "

            except (ValueError, TypeError, OverflowError):
                pass

        # Similarity-based categorical value if not too many distinct values
        if value is None and num_unique <= sqrt(num_data):  # num_data must be > 161 to be true.
            if _column_does_not_contain_genuine_floats(col):
                if num_unique > 2:  # note the alternative is never possible anyway.
                    true_categorical = not _is_numeric(col)
                    value = CategoricalMeta(name, similarity_based=True, true_categorical=true_categorical)
                    reason = "Small (< sqrt(N)) number of distinct values. "

        # Return non-numeric value and handle NaNs if necessary
        if value is not None:
            return value, reason

        # ========== Numeric value ==========
        # Try parsing if object type
        if col.dtype.kind == 'O':
            numeric_data = pd.to_numeric(col, errors='coerce')
            num_nan = numeric_data.isna().sum()
            if num_nan / num_data < self.config.parsing_nan_fraction_threshold:
                assert numeric_data.dtype.kind in ('f', 'i')
            else:
                numeric_data = None
        elif col.dtype.kind in ('f', 'i'):
            numeric_data = col

        else:
            numeric_data = None

        # Return numeric value and handle NaNs if necessary
        if numeric_data is not None and numeric_data.dtype.kind in ('f', 'i'):
            value = ContinuousMeta(name)
            reason = f"Converted to numeric dtype ({numeric_data.dtype.kind}) with success " + \
                     f"rate > {1.0 - self.config.parsing_nan_fraction_threshold}. "

            is_nan = numeric_data.isna().any()
            is_inf = numeric_data.isin([np.Inf, -np.Inf]).any()

            if is_nan or is_inf:
                value = NanMeta(name, value)
                reason += " And contains "
                reason += 'NaNs/Infs. ' if is_nan and is_inf else 'NaNs. ' if is_nan else 'Infs. '

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


def _is_numeric(col: pd.Series) -> bool:
    """Check whether col contains only numeric values"""
    if col.dtype.kind in ('f', 'i', 'u'):
        return True
    elif col.astype(str).str.isnumeric().all():
        return True
    else:
        return False
