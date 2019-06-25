from typing import List

import pandas as pd

from .rule import RuleValue
from .value import Value


def identify_rules(df: pd.DataFrame, values: List[Value]) -> List[Value]:
    return values

    # Bundle first three values into rule value
    value = RuleValue(name='rule1', values=values[:3], function='pick-first')

    # Replace first three values with rule value
    return [value] + values[3:]
