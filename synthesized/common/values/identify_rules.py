from typing import List

import pandas as pd

from .rule import RuleValue
from .value import Value
from .continuous import ContinuousValue
from .categorical import CategoricalValue
from .nan import NanValue


def identify_rules(df: pd.DataFrame, values: List[Value]) -> List[Value]:
    """ Loops through all pairs of values and finds the presence of simple
    rules.
    """
    #  return values
    # Find pairwise relationships
    N = len(values)
    for i in range(N):
        for j in range(i+1, N):
            if isinstance(values[i], ContinuousValue):
                if isinstance(values[j], CategoricalValue):
                    # Find if threshold related. E.g. age and retirement
                    pass
                elif isinstance(values[j], ContinuousValue):
                    # Find if one is a linear function of the other. E.g. celcius and farenheit
                    pass
                elif isintance(values[j], NanValue):
                    pass
            elif isinstance(values[i], CategoricalValue):
                if isinstance(values[j], CategoricalValue):
                    pass
                elif isinstance(values[j], ContinuousValue):
                    pass
                elif isintance(values[j], NanValue):
                    pass
            elif isinstance(values[i], NanValue):
                if isinstance(values[j], CategoricalValue):
                    pass
                elif isinstance(values[j], ContinuousValue):
                    pass
                elif isintance(values[j], NanValue):
                    pass

    # Bundle first three values into rule value
    #  value = RuleValue(name='rule1', values=values[:3], function='pick-first')

    # Replace first three values with rule value
    #  return [value] + values[3:]
    return values
