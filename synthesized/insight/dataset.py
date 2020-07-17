from itertools import chain
from typing import List, Tuple, Union

import pandas as pd

from ..metadata import MetaExtractor, DataFrameMeta
from ..metadata import ContinuousMeta, CategoricalMeta, DecomposedContinuousMeta, NanMeta, ValueMeta, AssociationMeta


def describe_dataset_values(df: pd.DataFrame) -> pd.DataFrame:
    dp = MetaExtractor.extract(df=df)
    values = dp.values

    value_spec = [
        {k: j for k, j in chain(v.specification().items(), [('class_name', v.__class__.__name__)])}
        for v in values
    ]
    for s in value_spec:
        if 'categories' in s:
            s['categories'] = '[' + ', '.join([str(o) for o in s['categories']]) + ']'

    for n, v in enumerate(values):
        if isinstance(v, NanMeta):
            value_spec.append(
                {'class_name': v.value.__class__.__name__, 'name': v.name + '_value'})

    df_values = pd.DataFrame.from_records(value_spec)

    return df_values


def categorical_or_continuous_values(df_or_dp: Union[pd.DataFrame, DataFrameMeta]) \
        -> Tuple[List[CategoricalMeta], List[ValueMeta]]:
    dp = MetaExtractor.extract(df=df_or_dp) if isinstance(df_or_dp, pd.DataFrame) else df_or_dp
    values = dp.values
    categorical: List[CategoricalMeta] = list()
    continuous: List[ValueMeta] = list()

    for value in values:
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
        elif isinstance(value, ContinuousMeta) or isinstance(value, DecomposedContinuousMeta):
            continuous.append(value)
        elif isinstance(value, NanMeta):
            if isinstance(value.value, ContinuousMeta) or isinstance(value.value, DecomposedContinuousMeta):
                continuous.append(value)

    return categorical, continuous


def describe_dataset(df: pd.DataFrame) -> pd.DataFrame:
    value_counts = describe_dataset_values(df).groupby('class_name').size().to_dict()

    properties = {f'num_{value}': count for value, count in value_counts.items()}
    properties['total_rows'] = len(df)
    properties['total_columns'] = sum([n for v, n in value_counts.items() if v != 'NanValue'])

    return pd.DataFrame.from_records([properties]).T.reset_index().rename(columns={'index': 'property', 0: 'value'})


def format_time_series(df, identifier, time_index):
    if time_index is not None:
        df[time_index] = pd.to_datetime(df[time_index])

    if identifier is not None:
        df = df.pivot(index=time_index, columns=identifier)
        df = df.swaplevel(0, 1, axis=1)
    return df.asfreq(infer_freq(df))


def infer_freq(df):
    freqs = [
        'B', 'D', 'W', 'M', 'SM', 'BM', 'MS', 'SMS', 'BMS', 'Q', 'BQ', 'QS', 'BQS', 'A', 'Y', 'BA', ' BY', 'AS', 'YS',
        'BAS', 'BYS', 'BH', 'H'
    ]
    best, min = None, None
    for freq in freqs:
        df2 = df.asfreq(freq)
        if len(df2) < len(df):
            continue
        nan_count = sum(df2.iloc[:, 0].isna())
        if min is None or nan_count < min:
            min = nan_count
            best = freq
    print(best)
    return best
