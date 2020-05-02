from itertools import chain
from typing import List, Tuple, Union

import pandas as pd

from ..common import ValueFactory
from ..common.values import ContinuousValue, CategoricalValue, NanValue


def describe_dataset_values(df: pd.DataFrame) -> pd.DataFrame:
    vf = ValueFactory(df=df)
    values = vf.get_values()

    value_spec = [
        {k: j for k, j in chain(v.specification().items(), [('class_name', v.__class__.__name__)])}
        for v in values
    ]
    for s in value_spec:
        if 'categories' in s:
            s['categories'] = '[' + ', '.join([str(o) for o in s['categories']]) + ']'

    for n, v in enumerate(values):
        if hasattr(v, 'day'):
            value_spec[n]['embedding_size'] = v.learned_input_size()
        if v.__class__.__name__ == 'NanValue':
            value_spec.append(
                {'class_name': v.value.__class__.__name__, 'name': v.name + '_value'})

    df_values = pd.DataFrame.from_records(value_spec)

    return df_values


def categorical_or_continuous_values(df_or_vf: Union[pd.DataFrame, ValueFactory]) -> Tuple[List[str], List[str]]:
    vf = ValueFactory(df=df_or_vf) if isinstance(df_or_vf, pd.DataFrame) else df_or_vf

    values = vf.get_values()
    categorical, continuous = [], []

    for value in values:
        if isinstance(value, CategoricalValue):
            if value.true_categorical:
                categorical.append(value.name)
            else:
                continuous.append(value.name)
        elif isinstance(value, ContinuousValue) or isinstance(value, NanValue):
            continuous.append(value.name)

    return categorical, continuous


def describe_dataset(df: pd.DataFrame) -> pd.DataFrame:
    value_counts = describe_dataset_values(df).groupby('class_name').size().to_dict()

    properties = {f'num_{value}': count for value, count in value_counts.items()}
    properties['total_rows'] = len(df)
    properties['total_columns'] = sum([n for v, n in value_counts.items() if v != 'NanValue'])

    return pd.DataFrame.from_records([properties]).T.reset_index().rename(columns={'index': 'property', 0: 'value'})
