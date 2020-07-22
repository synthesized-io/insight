from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union, Tupl, Sequence

import numpy as np
import pandas as pd

from .association import AssociationMeta
from .continuous import ContinuousMeta
from .categorical import CategoricalMeta
from .date import DateMeta
from .decomposed_continuous import DecomposedContinuousMeta
from .nan import NanMeta
from .value_meta import ValueMeta
from .date import TimeIndexMeta
from .identifier import IdentifierMeta


class DataFrameMeta:
    """A smart container for the various types of data sets."""
    def __init__(
            self, values: List[ValueMeta], id_value: Optional[IdentifierMeta] = None,
            time_value: Optional[TimeIndexMeta] = None,
            column_aliases: Dict[str, str] = None, association_meta: AssociationMeta = None
    ):
        self.values = values
        self.id_value = id_value
        self.time_value = time_value
        self.columns = [col for value in self.values for col in value.columns()]
        self.id_index_name = id_value.name if id_value is not None else None
        self.time_index_name = time_value.name if time_value is not None else None
        self.column_aliases = column_aliases or dict()
        self.association_meta = association_meta

        self._value_map = self.compute_value_map()

    def compute_value_map(self) -> Dict[str, ValueMeta]:
        value_map: Dict[str, ValueMeta] = {v.name: v for v in self.values}
        if self.time_value is not None:
            value_map[self.time_value.name] = self.time_value
        if self.id_value is not None:
            value_map[self.id_value.name] = self.id_value
            self.columns = [self.id_value.name, ] + self.columns

        return value_map

        self.categorical: Optional[List[CategoricalMeta]] = None
        self.continuous: Optional[List[ValueMeta]] = None

    @property
    def all_values(self) -> List[ValueMeta]:
        values = self.values

        if self.time_value:
            time_index_values: List[ValueMeta] = [self.time_value]
            values = time_index_values + values

        if self.id_value:
            id_index_values: List[ValueMeta] = [self.id_value]
            values = id_index_values + values

        return values

    @property
    def indices(self) -> List[Union[IdentifierMeta, TimeIndexMeta]]:
        indices: List[Union[IdentifierMeta, TimeIndexMeta]] = []

        if self.id_value:
            indices.append(self.id_value)
        if self.time_value:
            indices.append(self.time_value)

        return indices

    def set_indices(self, df: pd.DataFrame):
        for index in self.indices:
            index.set_index(df=df)

        df.sort_index(inplace=True)
        return df

    def __getitem__(self, item: str) -> ValueMeta:
        return self._value_map[item]

    def unified_inputs(self, inputs: Dict[str, np.ndarray]) -> Dict[str, List[np.ndarray]]:
        # Concatenate input tensors per value
        x = {
            vm.name: [inputs[col_name] for col_name in vm.learned_input_columns()]
            for vm in self.values
        }
        return x

    def split_outputs(self, outputs: Dict[str, Sequence[Any]]) -> Dict[str, np.ndarray]:
        # Concatenate input tensors per value
        values = self.values
        if self.id_value:
            values = values + [self.id_value]
        if self.time_value:
            values = values + [self.time_value]

        x = self.convert_tf_to_np_dict({
            col_name: outputs[vm.name][n]
            for vm in values
            for n, col_name in enumerate(
                vm.learned_output_columns() if not isinstance(vm, IdentifierMeta)
                else [vm.name]
            )
        })

        return x

    @staticmethod
    def convert_tf_to_np_dict(tf_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        for name, tensor in tf_dict.items():
            try:
                tf_dict[name] = tensor.numpy()
            except AttributeError:
                tf_dict[name] = tensor

        return tf_dict

    def preprocess(self, df: pd.DataFrame, max_workers: Optional[int] = 4) -> pd.DataFrame:
        """Returns a preprocessed copy of the input DataFrame"""
        if max_workers is None:
            df_pre = df.copy()
            for value in self.all_values:
                df_pre = value.preprocess(df=df_pre)

        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                arguments = ((value, df[value.columns()].copy()) for value in self.all_values)
                col_pre = executor.map(self.preprocess_value, arguments)

            series = []
            for v, f in zip(self.all_values, col_pre):
                series.append(f)
            df_pre = pd.concat(series, axis=1)

        return df_pre

    @staticmethod
    def preprocess_value(argument: Tuple[ValueMeta, pd.DataFrame]) -> pd.DataFrame:
        value, df_copy = argument
        df_copy = value.preprocess(df_copy)
        return df_copy

    def preprocess_by_name(self, df: Union[pd.DataFrame, None], value_names: List[str]) -> Union[pd.DataFrame, None]:
        """Returns a preprocessed copy of the input conditions dataframe"""
        if df is not None:
            if isinstance(df, dict):
                df_conditions = pd.DataFrame.from_dict(
                    {name: np.reshape(condition, (-1,)) for name, condition in df.items()}
                )
            else:
                df_conditions = df.copy()

            for name in value_names:
                df_conditions = self.__getitem__(name).preprocess(df=df_conditions)
        else:
            df_conditions = None

        return df_conditions

    def postprocess(self, df: pd.DataFrame, max_workers: Optional[int] = 4) -> pd.DataFrame:
        """Post-processes the input DataFrame"""
        df_post = df.copy()
        if max_workers is None:
            for value in self.all_values:
                df_post = value.postprocess(df=df_post)

        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                arguments = ((value, df[value.learned_output_columns()].copy()) for value in self.all_values)
                futures = executor.map(self.postprocess_value, arguments)

            series = []
            names = []
            for v, f in zip(self.all_values, futures):
                series.append(f)
                names.append(v.name)
            df_post = pd.concat(series, axis=1)

        # aliases:
        for alias, col in self.column_aliases.items():
            df_post[alias] = df_post[col]

        assert len(df_post.columns) == len(self.columns)
        df_post = df_post[self.columns]

        return df_post

    @staticmethod
    def postprocess_value(argument: Tuple[ValueMeta, pd.DataFrame]) -> pd.DataFrame:
        value, df_copy = argument
        df_copy = value.postprocess(df_copy)
        return df_copy

    def get_categorical(self):
        if self.categorical is not None:
            return self.categorical

        categorical, _ = self.get_categorical_and_continuous()
        return categorical

    def get_continuous(self):
        if self.continuous is not None:
            return self.continuous

        _, continuous = self.get_categorical_and_continuous()
        return continuous

    def get_categorical_and_continuous(self) -> Tuple[List[CategoricalMeta], List[ValueMeta]]:

        if self.categorical is not None and self.continuous is not None:
            return self.categorical, self.continuous

        categorical: List[CategoricalMeta] = list()
        continuous: List[ValueMeta] = list()

        for value in self.values:
            if isinstance(value, CategoricalMeta):
                if value.true_categorical:
                    categorical.append(value)
                else:
                    continuous.append(value)
            elif isinstance(value, AssociationMeta):
                for associated_value in value.values:
                    if associated_value.true_categorical:
                        categorical.append(associated_value)
                    else:
                        continuous.append(associated_value)
            elif isinstance(value, DateMeta):
                # To avoid DateMeta get into Continuous
                pass
            elif isinstance(value, ContinuousMeta) or isinstance(value, DecomposedContinuousMeta):
                continuous.append(value)
            elif isinstance(value, NanMeta):
                if isinstance(value.value, ContinuousMeta) or isinstance(value.value, DecomposedContinuousMeta):
                    continuous.append(value)

        self.categorical = categorical
        self.continuous = continuous

        return self.categorical, self.continuous

    def get_variables(self) -> Dict[str, Any]:
        variables: Dict[str, Any] = dict(
            columns=self.columns,
            column_aliases=self.column_aliases,
            id_index=self.id_index_name,
            identifier_value=self.id_value.get_variables() if self.id_value else None,
            time_index=self.time_index_name,
            time_value=self.time_value.get_variables() if self.time_value else None,
            association_meta=self.association_meta.get_variables() if self.association_meta is not None else None
        )

        variables['num_values'] = len(self.values)
        for i, value in enumerate(self.values):
            variables['value_{}'.format(i)] = value.get_variables()

        return variables

    def set_variables(self, variables: Dict[str, Any]):
        self.columns = variables['columns']
        self.column_aliases = variables['column_aliases']
        self.id_index_name = variables['id_index']
        self.id_value = ValueMeta.set_variables(variables['identifier_value']) \
            if variables['identifier_value'] is not None else None
        self.time_index_name = variables['id_index']
        self.time_value = ValueMeta.set_variables(variables['time_value']) \
            if variables['time_value'] is not None else None
        self.association_meta = ValueMeta.set_variables(variables['association_meta']) \
            if variables['time_value'] is not None else None

        self.values = []
        for i in range(variables['num_values']):
            self.values.append(ValueMeta.set_variables(variables['value_{}'.format(i)]))

        self._value_map = self.compute_value_map()

    @classmethod
    def from_dict(cls, variables: Dict[str, Any]) -> 'DataFrameMeta':
        df_meta = cls.__new__(cls)
        df_meta.set_variables(variables)
        return df_meta
