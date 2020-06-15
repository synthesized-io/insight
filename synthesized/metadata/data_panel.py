from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .value_meta import ValueMeta


class DataPanel:
    """A smart container for the various types of data sets."""
    def __init__(
            self, values: List[ValueMeta], id_value: Optional[ValueMeta] = None, time_value: Optional[ValueMeta] = None,
            column_aliases: Dict[str, str] = None
    ):
        self.values = values
        self.identifier_value = id_value
        self.time_value = time_value
        self.columns = [col for value in self.values for col in value.columns()]
        self.id_index = id_value.name if id_value is not None else None
        self.time_index = time_value.name if time_value is not None else None
        self.column_aliases = column_aliases or dict()

        value_map = {v.name: v for v in self.values}
        if time_value is not None:
            value_map[time_value.name] = time_value
        if id_value is not None:
            value_map[id_value.name] = id_value

        self._value_map = value_map

    @property
    def all_values(self):
        values = self.values

        if self.identifier_value:
            values = values + [self.identifier_value]

        if self.time_value:
            values = values + [self.time_value]

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
        print(outputs)
        x = {
            col_name: outputs[vm.name][n]
            for vm in self.values
            for n, col_name in enumerate(vm.learned_output_columns())
        }
        return x

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a preprocessed copy of the input DataFrame"""
        df_copy = df.copy()
        for value in self.all_values:
            df_copy = value.preprocess(df=df_copy)

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

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-processes the input DataFrame"""
        for value in self.all_values:
            df = value.postprocess(df=df)

        # aliases:
        for alias, col in self.column_aliases.items():
            df[alias] = df[col]

        assert len(df.columns) == len(self.columns)
        df = df[self.columns]

        return df

    def get_variables(self) -> Dict[str, Any]:
        variables: Dict[str, Any] = dict(
            columns=self.columns,
            id_index=self.id_index,
            identifier_value=self.identifier_value.get_variables() if self.identifier_value else None,
            time_index=self.time_index,
            time_value=self.time_value.get_variables() if self.time_value else None
        )

        variables['num_values'] = len(self.values)
        for i, value in enumerate(self.values):
            variables['value_{}'.format(i)] = value.get_variables()

        return variables

    def set_variables(self, variables: Dict[str, Any]):
        self.columns = variables['columns']
        self.id_index = variables['id_index']
        self.identifier_value = ValueMeta.set_variables(variables['identifier_value']) \
            if variables['identifier_value'] is not None else None
        self.time_index = variables['id_index']
        self.time_value = ValueMeta.set_variables(variables['time_value']) \
            if variables['time_value'] is not None else None

        self.values = []
        for i in range(variables['num_values']):
            self.values.append(ValueMeta.set_variables(variables['value_{}'.format(i)]))

    @classmethod
    def from_dict(cls, variables: Dict[str, Any]) -> 'DataPanel':
        data_panel = cls.__new__(cls)
        data_panel.set_variables(variables)
        return data_panel
