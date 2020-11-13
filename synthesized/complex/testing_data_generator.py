from typing import List

import pandas as pd
import yaml

from ..metadata import ValueMeta, DataFrameMeta


class TestingDataGenerator:
    def __init__(self, df_meta: DataFrameMeta):
        self.df_meta = df_meta

    @classmethod
    def from_dict(cls, kwargs):
        value_meta_dict = {vm.__name__.lower()[:-4]: vm for vm in ValueMeta.__subclasses__()}

        values = []
        for value_kwargs in kwargs:
            value_type = value_kwargs['type']
            value_cls = value_meta_dict[value_type]

            value = value_cls.from_dict(value_kwargs)
            if len(value.learned_output_columns()) > 0:
                raise ValueError(f"TestingDataGenerator not compatible with '{value_type}'")

            values.append(value)

        df_meta = DataFrameMeta(values)
        return cls(df_meta)

    def synthesize(self, num_rows: int) -> pd.DataFrame:
        return self.df_meta.postprocess(pd.DataFrame([[], ] * num_rows))

    @classmethod
    def generate_data_from_yaml(cls, file_name):

        with open(file_name, 'r') as fp:
            kwargs = yaml.load(fp)

        num_rows, output_file, values = cls.validate_arguments(["num_rows", "output_file", "values"], kwargs)
        generator = cls.from_dict(values)

        df_synthesized = generator.synthesize(num_rows)
        output_format = kwargs.get("output_format", "csv")
        if output_format == "csv":
            df_synthesized.to_csv(output_file)
        elif output_format == "json":
            df_synthesized.to_json(output_file, orient='records', indent=2)

    @staticmethod
    def validate_arguments(args, kwargs) -> List:
        values = []
        for arg in args:
            if arg not in kwargs:
                raise ValueError(f"Key '{arg}' not found in given file.")
            values.append(kwargs[arg])

        return values
