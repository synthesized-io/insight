from collections import Counter
import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from ..metadata import ValueMeta, DataFrameMeta
from ..util import get_all_subclasses

logger = logging.getLogger(__name__)


class TestingDataGenerator:
    def __init__(self, df_meta: DataFrameMeta):
        self.df_meta = df_meta

    @classmethod
    def from_dict(cls, value_kwargs: Optional[List[Dict[str, Any]]],
                  groups_kwargs: Optional[List[Dict[str, Any]]]) -> 'TestingDataGenerator':

        logger.info("Initializing Testing Data Generator.")
        value_meta_dict = {vm.__name__.lower()[:-4]: vm for vm in get_all_subclasses(ValueMeta)}

        values = []
        # Values
        if value_kwargs:
            for value_kwargs_i in value_kwargs:
                value_type = value_kwargs_i['type']
                value_cls = value_meta_dict[value_type]

                value = value_cls.from_dict(value_kwargs_i)
                if len(value.learned_output_columns()) > 0:
                    raise ValueError(f"TestingDataGenerator not compatible with '{value_type}'")

                values.append(value)

        # Groups (various values with same parameters)
        if groups_kwargs:
            for group_kwargs_i in groups_kwargs:
                group_type = group_kwargs_i['type']
                value_cls = value_meta_dict[group_type]

                group_name = group_kwargs_i.pop('name', None)
                group_count = group_kwargs_i.pop('count', None)
                group_names = group_kwargs_i.pop('names', None)

                # Given 'name_format' and 'count'
                if group_name and group_count:
                    names = [group_name.format(str(i)) for i in range(group_count)]

                # Given 'names'
                elif group_names:
                    names = group_names

                else:
                    raise ValueError("Both 'group_name' and 'group_count', or 'group_names' must be given")

                for name in names:
                    value = value_cls.from_dict(group_kwargs_i, name=name)
                    if len(value.learned_output_columns()) > 0:
                        raise ValueError(f"TestingDataGenerator not compatible with '{group_type}'")

                    values.append(value)

        all_names = [value.name for value in values]
        duplicated_names = [name for name, counts in Counter(all_names).items() if counts > 1]
        if len(duplicated_names) > 0:
            raise ValueError(f"All names must be unique. Given {', '.join(duplicated_names)} are duplicated.")

        logger.info("Extracting meta-information from given rules.")
        df_meta = DataFrameMeta(values=values)
        return cls(df_meta)

    @classmethod
    def from_yaml(cls, config_file_name: str):

        logger.info("Reading configuration file.")
        with open(config_file_name, 'r') as fp:
            kwargs = yaml.safe_load(fp)

        values = kwargs.get("values", None)
        groups = kwargs.get("groups", {})
        generator = cls.from_dict(values, groups)

        return generator

    def synthesize(self, num_rows: int) -> pd.DataFrame:
        logger.info("Generating data.")
        return self.df_meta.postprocess(pd.DataFrame([[], ] * num_rows))

    @staticmethod
    def validate_arguments(args: List[str], kwargs: Dict[str, Any]) -> List:
        values = []
        for arg in args:
            if arg not in kwargs:
                raise ValueError(f"Key '{arg}' not found in given file.")
            values.append(kwargs[arg])

        return values
