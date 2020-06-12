from typing import Any, Dict, List, Optional, Union

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

    @property
    def all_values(self):
        values = self.values

        if self.identifier_value:
            values = values + [self.identifier_value]

        if self.time_value:
            values = values + [self.time_value]

        return values

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns a preprocessed copy of the input DataFrame"""
        df_copy = df.copy()
        for value in self.all_values:
            df_copy = value.preprocess(df=df_copy)

        return df_copy

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
