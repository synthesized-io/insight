from typing import List, Optional

import pandas as pd

from .rule import RuleValue
from .value import Value
from .continuous import ContinuousValue
from .categorical import CategoricalValue

# When finding separable values, want to limit the maximum number of groups to search over. A categorical variable with
# many categories will likely split continuous variables neatly.
MAX_CATEGORIES_FOR_THRESH = 2


def argsort(l):
    return sorted(range(len(l)), key=l.__getitem__)


def find_piecewise(df: pd.DataFrame, x: ContinuousValue, y: CategoricalValue) -> Optional[RuleValue]:
    """ Splits the data into sets for each category. If each set is non-overlapping, then we have a flag
    variable.
    """
    if len(y.categories) > MAX_CATEGORIES_FOR_THRESH:
        return

    sets = [df.loc[df[y.columns()[0]] == v, x.columns()[0]] for v in y.categories]
    categories = [v for v in y.categories]
    maxes = [s.max() for s in sets]
    mins = [s.min() for s in sets]

    # Sort the set by their maximum value
    idx = argsort(maxes)
    maxes = [maxes[i] for i in idx]
    mins = [mins[i] for i in idx]
    categories = [categories[i] for i in idx]

    for i in range(len(idx)-1):
        if maxes[i] >= mins[i + 1]:
            return

    # Make the threshold the midway between the two boundaries
    threshs = [(lower_max + upper_min)/2 for lower_max, upper_min in zip(maxes[:-1], mins[1:])]

    return RuleValue(name=x.name + '_' + y.name, values=[x, y], function='flag_1',
                     fkwargs=dict(threshs=threshs, categories=categories))


@profile
def find_pulse(df: pd.DataFrame, x: ContinuousValue, y: CategoricalValue) -> Optional[RuleValue]:
    """ Checks if a categorical variable with 2 options splits the continuous variable into two regions, one
    completely inside the other and with no overlap. E.g. 'working age' vs 'age' would satisfy this.
    """
    if len(y.categories) > 2:
        return

    set1 = df.loc[df[y.columns()[0]] == y.categories[0], x.columns()[0]]
    set2 = df.loc[df[y.columns()[0]] == y.categories[1], x.columns()[0]]

    if set1.max() > set2.max() and set1.min() < set2.min():
        pass  # set1 is already the outer set
    elif set2.max() > set1.max() and set2.min() < set1.min():
        temp = set1
        set1 = set2
        set2 = temp
    else:
        return

    # If all the mass of set1 is completely outside set2, then we have a pulse
    Nlower = len(set1 < set2.min())
    Nupper = len(set1 > set2.max())
    if Nlower + Nupper == len(set1):
        return RuleValue(name=x.name + '_' + y.name, values=[x, y], function='pulse',
                         fkwargs=dict(lower=set2.min(), upper=set2.max()))


def identify_rules(df: pd.DataFrame, values: List[Value]) -> List[Value]:
    """ Loops through all pairs of values and finds the presence of simple
    rules.
    """
    # Define the set of tests to do for pairs of variables
    continuous_categorical_tests = [find_piecewise, find_pulse]

    # Find pairwise relationships
    M = len(values)
    new_values = []
    base_vars = [False,] * M
    derived_vars = [False,] * M
    for i in range(M):
        if derived_vars[i] or base_vars[i]:
            continue

        rule = None
        for j in range(i+1, M):
            # Detect the type of rule tests to run
            tests = []
            if isinstance(values[i], ContinuousValue):
                if isinstance(values[j], CategoricalValue):
                    tests = continuous_categorical_tests
                    base = i
                    derived = j
                elif isinstance(values[j], ContinuousValue):
                    pass
            elif isinstance(values[i], CategoricalValue):
                if isinstance(values[j], CategoricalValue):
                    pass
                elif isinstance(values[j], ContinuousValue):
                    tests = continuous_categorical_tests
                    base = j
                    derived = i

            # Perform all tests, stopping if one has been successful
            for test in tests:
                rule = test(df, values[base], values[derived])
                if rule is not None:
                    break

            # If we have found a rule, add it to the list of values and move to the next value
            if rule is not None:
                base_vars[base] = True
                derived_vars[derived] = True
                new_values.append(rule)
                break
        else:
            # No rules were found
            new_values.append(values[i])

    # Bundle first three values into rule value
    #  value = RuleValue(name='rule1', values=values[:3], function='pick-first')

    # Replace first three values with rule value
    #  return [value] + values[3:]
    return new_values
