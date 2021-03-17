from itertools import chain

import pandas as pd

from ..metadata_new.factory import MetaExtractor


def describe_dataset_values(df: pd.DataFrame) -> pd.DataFrame:
    dp = MetaExtractor.extract(df=df)

    meta_dict = [
        {k: j for k, j in chain(vm.to_dict().items(), [('class_name', vm.__class__.__name__)])}
        for vm in dp.values()
    ]
    for s in meta_dict:
        if 'categories' in s:
            assert isinstance(s['categories'], list)
            s['categories'] = '[' + ', '.join([str(o) for o in s['categories']]) + ']'

    df_values = pd.DataFrame.from_records(meta_dict)

    return df_values


def describe_dataset(df: pd.DataFrame) -> pd.DataFrame:
    value_counts = describe_dataset_values(df).groupby('class_name').size().to_dict()

    properties = {f'num_{value}': count for value, count in value_counts.items()}
    properties['total_rows'] = len(df)
    properties['total_columns'] = sum([n for n in value_counts.values()])

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
