from itertools import chain

import pandas as pd

from ..common import ValueFactory


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
