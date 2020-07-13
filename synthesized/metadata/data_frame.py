from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .association import AssociationMeta
from .value_meta import ValueMeta


class DataFrameMeta:
    """A smart container for the various types of data sets."""
    def __init__(
            self, values: List[ValueMeta], id_value: Optional[ValueMeta] = None, time_value: Optional[ValueMeta] = None,
            column_aliases: Dict[str, str] = None, association_meta: AssociationMeta = None
    ):
        self.values = values
        self.identifier_value = id_value
        self.time_value = time_value
        self.columns = [col for value in self.values for col in value.columns()]
        self.id_index = id_value.name if id_value is not None else None
        self.time_index = time_value.name if time_value is not None else None
        self.column_aliases = column_aliases or dict()
        self.association_meta = association_meta

        value_map = {v.name: v for v in self.values}
        if time_value is not None:
            value_map[time_value.name] = time_value
            self.columns = [time_value.name, ] + self.columns
        if id_value is not None:
            value_map[id_value.name] = id_value
            self.columns = [id_value.name, ] + self.columns

        self._value_map = value_map

    @property
    def all_values(self):
        values = self.values

        if self.time_value:
            values = [self.time_value] + values

        if self.identifier_value:
            values = [self.identifier_value] + values

        return values

    def __getitem__(self, item):
        return self._value_map[item]

    def unified_inputs(self, inputs: Dict[str, np.ndarray]) -> Dict[str, List[np.ndarray]]:
        # Concatenate input tensors per value
        x = {
            vm.name: [inputs[col_name] for col_name in vm.learned_input_columns()]
            for vm in self.values
        }
        return x

    def split_outputs(self, outputs: Dict[str, List]) -> Dict[str, Any]:
        # Concatenate input tensors per value
        values = self.values
        if self.identifier_value:
            values = values + [self.identifier_value]
        if self.time_value:
            values = values + [self.time_value]

        x = {
            col_name: outputs[vm.name][n]
            for vm in values
            for n, col_name in enumerate(vm.learned_output_columns())
        }

        return x

    def preprocess(self, df: pd.DataFrame, max_workers: Optional[int] = 4) -> pd.DataFrame:
        """Returns a preprocessed copy of the input DataFrame"""
        if max_workers is None:
            df_pre = df.copy()
            for value in self.all_values:
                df_pre = value.preprocess(df=df_pre)

        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                arguments = ((value, df[[value.name]].copy()) for value in self.all_values)
                col_pre = executor.map(self.preprocess_value, arguments)

            series = []
            for v, f in zip(self.all_values, col_pre):
                series.append(f)
            df_pre = pd.concat(col_pre, axis=1)

        return df_pre

    @staticmethod
    def preprocess_value(argument):
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
                arguments = ((value, df[[value.name]].copy()) for value in self.all_values)
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

        assert len(df.columns) == len(self.columns)
        df_post = df_post[self.columns]

        return df_post

    @staticmethod
    def postprocess_value(argument):
        value, df_copy = argument
        df_copy = value.postprocess(df_copy)
        return df_copy

    def get_variables(self) -> Dict[str, Any]:
        variables: Dict[str, Any] = dict(
            columns=self.columns,
            column_aliases=self.column_aliases,
            id_index=self.id_index,
            identifier_value=self.identifier_value.get_variables() if self.identifier_value else None,
            time_index=self.time_index,
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
        self.id_index = variables['id_index']
        self.identifier_value = ValueMeta.set_variables(variables['identifier_value']) \
            if variables['identifier_value'] is not None else None
        self.time_index = variables['id_index']
        self.time_value = ValueMeta.set_variables(variables['time_value']) \
            if variables['time_value'] is not None else None
        self.association_meta = ValueMeta.set_variables(variables['association_meta']) \
            if variables['time_value'] is not None else None

        self.values = []
        for i in range(variables['num_values']):
            self.values.append(ValueMeta.set_variables(variables['value_{}'.format(i)]))

    @classmethod
    def from_dict(cls, variables: Dict[str, Any]) -> 'DataFrameMeta':
        df_meta = cls.__new__(cls)
        df_meta.set_variables(variables)
        return df_meta
