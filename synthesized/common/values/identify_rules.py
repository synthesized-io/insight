import pandas as pd

from .rule import RuleValue


def identify_rules(df, values):
    return values

    value = RuleValue(name='rule1', values=values[:3], function='pick-first')
    return [value] + values[3:]
