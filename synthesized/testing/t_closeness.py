from itertools import combinations
from operator import itemgetter

from pyemd import emd_samples


def max_emd(df1, df2, column_names):
    distances = []
    for column in column_names:
        a1 = df1[column]
        a2 = df2[column]
        dist = emd_samples(a1, a2)
        distances.append((column, dist))
    return max(distances, key=itemgetter(1))


def t_closeness_check(df, threshold=0.2):
    """
    Returns a list of dataframes where each dataframe represents attributes that can be used
    to find equivalence classes which do not satisfy t-closeness requirement

    """

    def is_t_close(group, columns):
        column, distance = max_emd(df, group, columns)
        if distance < threshold:
            return True
        else:
            return False

    result = []
    columns = set(df.columns.values)
    for i in range(1, len(columns) + 1):
        for subset in combinations(columns, i):
            sensitive_columns = columns - set(subset)
            if len(sensitive_columns) == 0:
                continue
            vulnerable_rows = df.groupby(by=list(subset)).filter(
                lambda g: not is_t_close(g, columns=sensitive_columns))
            vulnerable_identifiers = vulnerable_rows.drop(sensitive_columns, axis=1)
            if not vulnerable_identifiers.empty:
                result.append(vulnerable_identifiers)
    return result
