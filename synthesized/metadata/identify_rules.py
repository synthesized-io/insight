from typing import List, Union, Optional, cast

import pandas as pd

from .categorical import CategoricalMeta
from .continuous import ContinuousMeta
from .rule import RuleMeta
from .value_meta import ValueMeta

# When finding separable values, want to limit the maximum number of groups to search over. A categorical variable with
# many categories will likely split continuous variables neatly.
MAX_CATEGORIES_FOR_THRESH = 3
CONTINUOUS_VAR_SAMPLE = 1000


def _argsort(lst):
    # Find the indices that would sort a list
    return sorted(range(len(lst)), key=lst.__getitem__)


def _intersection(l1, l2):
    # Find the intersection of two lists
    return list(set(l1) & set(l2))


class PairwiseRuleFactory(object):
    """ Run Tests to search for pairwise relationship.

    The user must specify which tests they would like to run when building the synthesizer. This is the role of the
    'tests' variable in the find_relationship method. If this is 'all', all tests will be run, otherwise it should be a
    list of the names of individual tests, or the names of the groups of tests to run. For example:

        - 'all'
        - 'continuous_categorical'
        - ['find_piecewise', 'find_permute']
        - ['continuous_categorical', 'categorical_categorical']
        - ['continuous_categorical', 'find_permute']

    are all valid values.
    """
    continuous_categorical_tests = ['find_piecewise', 'find_pulse']
    continuous_continuous_tests = ['find_linear']
    categorical_categorical_tests = ['find_permute']

    @staticmethod
    def find_relationship(df: pd.DataFrame, x: ValueMeta, y: ValueMeta, tests: Union[str, List[str]]):
        rule = None
        if isinstance(x, ContinuousMeta) and isinstance(y, CategoricalMeta):
            rule = PairwiseRuleFactory.continuous_categorical(df, x, y, tests)
        elif isinstance(x, ContinuousMeta) and isinstance(y, ContinuousMeta):
            rule = PairwiseRuleFactory.continuous_continuous(df, x, y, tests)
        elif isinstance(x, CategoricalMeta) and isinstance(y, ContinuousMeta):
            rule = PairwiseRuleFactory.continuous_categorical(df, y, x, tests)
        elif isinstance(x, CategoricalMeta) and isinstance(y, CategoricalMeta):
            rule = PairwiseRuleFactory.categorical_categorical(df, y, x, tests)
        return rule

    @staticmethod
    def continuous_categorical(df: pd.DataFrame, x: ContinuousMeta, y: CategoricalMeta,
                               tests: Union[str, List[str]]) -> Optional[RuleMeta]:
        if tests == 'all' or tests == 'continuous_categorical':
            tests = PairwiseRuleFactory.continuous_categorical_tests
        elif isinstance(tests, list):
            if 'continuous_categorical' in tests:
                tests = PairwiseRuleFactory.continuous_categorical_tests
            else:
                tests = _intersection(tests, PairwiseRuleFactory.continuous_categorical_tests)
        else:
            return None

        # Do time consuming tasks here to save repeating in each test
        # TODO: what if y.categories is int?
        assert isinstance(y.categories, list)
        sets = [df.loc[df[y.columns()[0]] == v, x.columns()[0]] for v in y.categories]
        maxes = [s.max() for s in sets]
        mins = [s.min() for s in sets]
        rule = None
        if 'find_piecewise' in tests and rule is None:
            rule = PairwiseRuleFactory.find_piecewise(x, y, mins, maxes)
        if 'find_pulse' in tests and rule is None:
            rule = PairwiseRuleFactory.find_pulse(x, y, sets, mins, maxes)
        return rule

    @staticmethod
    def find_piecewise(x: ContinuousMeta, y: CategoricalMeta, mins: List[float],
                       maxes: List[float]) -> Optional[RuleMeta]:
        """ Splits the data into sets for each category. If each set is non-overlapping, then we have a
        piecewise constant relationship or 'flag' variable.
        """
        # TODO: what if y.categories is int?
        assert isinstance(y.categories, list)
        categories = [v for v in y.categories]
        if len(categories) > MAX_CATEGORIES_FOR_THRESH:
            return None

        # Sort the set by their maximum value
        idx = _argsort(maxes)
        maxes = [maxes[i] for i in idx]
        mins = [mins[i] for i in idx]
        categories = [categories[i] for i in idx]

        for i in range(len(idx)-1):
            if maxes[i] >= mins[i + 1]:
                return None

        # Make the threshold the midway between the two boundaries
        threshs = [(lower_max + upper_min)/2 for lower_max, upper_min in zip(maxes[:-1], mins[1:])]

        return RuleMeta(name=str(x.name) + '_' + str(y.name), values=[x, y], function='flag_1',
                        fkwargs=dict(threshs=threshs, categories=categories))

    @staticmethod
    def find_pulse(x: ContinuousMeta, y: CategoricalMeta, sets: List[pd.DataFrame], mins: List[float],
                   maxes: List[float]) -> Optional[RuleMeta]:
        """ Checks if a categorical variable with 2 options splits the continuous variable into two regions, one
        completely inside the other and with no overlap. E.g. 'working age' vs 'age' would satisfy this.
        """
        # TODO: what if y.categories is int?
        assert isinstance(y.categories, list)
        categories = [v for v in y.categories]
        if len(categories) > 2:
            return None

        # Make set1 the outer set or return
        if maxes[0] > maxes[1] and mins[0] < mins[1]:
            pass
        elif maxes[1] > maxes[0] and mins[1] < mins[0]:
            mins = mins[::-1]
            maxes = maxes[::-1]
            sets = sets[::-1]
            categories = categories[::-1]
        else:
            return None

        # If all the mass of set1 is completely outside set2, then we have a pulse
        Nlower = (sets[0] < mins[1]).sum()
        Nupper = (sets[0] > maxes[1]).sum()
        if Nlower + Nupper == len(sets[0]):
            threshs = dict(lower=mins[1], upper=maxes[1])
            return RuleMeta(name=str(x.name) + '_' + str(y.name), values=[x, y], function='pulse_1',
                            fkwargs=dict(threshs=threshs, categories=categories))
        else:
            return None

    @staticmethod
    def continuous_continuous(df: pd.DataFrame, x: ContinuousMeta, y: ContinuousMeta,
                              tests: Union[str, List[str]]) -> Optional[RuleMeta]:
        if tests == 'all' or tests == 'continuous_continuous':
            tests = PairwiseRuleFactory.continuous_continuous_tests
        elif isinstance(tests, list):
            if 'continuous_continuous' in tests:
                tests = PairwiseRuleFactory.continuous_continuous_tests
            else:
                tests = _intersection(tests, PairwiseRuleFactory.continuous_continuous_tests)
        else:
            return None

        rule = None
        if 'find_line' in tests and rule is None:
            # TODO: `cast` was added to pass mypy, however this call seems to be wrong
            rule = PairwiseRuleFactory.find_piecewise(x, cast(CategoricalMeta, y), [], [])

        return rule

    @staticmethod
    def find_line(df: pd.DataFrame, x: ContinuousMeta, y: ContinuousMeta) -> Optional[RuleMeta]:
        """ Checks if two continuous variables are linear functions of each other. Does this by finding
        gradient and intercept for a subset. If found, tests on all the data.
        """
        return None

    @staticmethod
    def categorical_categorical(df: pd.DataFrame, x: CategoricalMeta, y: CategoricalMeta,
                                tests: Union[str, List[str]]) -> Optional[RuleMeta]:
        if tests == 'all' or tests == 'categorical_categorical':
            tests = PairwiseRuleFactory.categorical_categorical_tests
        elif isinstance(tests, list):
            if 'categorical_categorical' in tests:
                tests = PairwiseRuleFactory.categorical_categorical_tests
            else:
                tests = _intersection(tests, PairwiseRuleFactory.categorical_categorical_tests)
        else:
            return None

        rule = None
        if 'find_permute' in tests and rule is None:
            rule = PairwiseRuleFactory.find_permute(x, y)

        return rule

    @staticmethod
    def find_permute(x: CategoricalMeta, y: CategoricalMeta) -> Optional[RuleMeta]:
        """ Checks if two categorical values are permutations of one another. """
        return None


def identify_rules(df: pd.DataFrame, values: List[ValueMeta], tests: Union[str, List[str]]) -> List[ValueMeta]:
    """ Loops through all pairs of values and finds the presence of simple
    rules.
    """
    if not tests:
        return values

    # Find pairwise relationships
    M = len(values)
    new_values = []
    base_vars = [False, ] * M
    derived_vars = [False, ] * M
    for i in range(M):
        if derived_vars[i] or base_vars[i]:
            continue

        rule = None
        for j in range(i+1, M):
            rule = PairwiseRuleFactory.find_relationship(df, values[i], values[j], tests)

            # If we have found a rule, add it to the list of values and move to the next value
            if rule is not None:
                if values[i] == rule.values[0]:
                    base_vars[i] = True
                    derived_vars[j] = True
                else:
                    base_vars[j] = True
                    derived_vars[i] = True
                new_values.append(rule)
                break
        else:
            # No rules were found
            new_values.append(values[i])

    # Bundle first three values into rule value
    #  value = RuleMeta(name='rule1', values=values[:3], function='pick-first')

    # Replace first three values with rule value
    #  return [value] + values[3:]
    return new_values
