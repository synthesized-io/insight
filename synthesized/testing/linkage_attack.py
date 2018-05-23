from itertools import combinations

import pandas as pd
from pyemd import emd_samples


def linkage_attack(df_orig, df_synth, categ_columns, t_closeness=0.2, k_distance=0.1):
    """categorical columns are not supported yet"""
    columns = set(df_orig.columns.values)
    result = []
    for attrs in t_closeness_check(df_orig, t_closeness):
        eq_class_orig = find_eq_class(df_orig, attrs)

        down, up = find_neighbour_distances(df_orig, attrs, categ_columns)
        eq_class_synth = find_eq_class_fuzzy(df_synth, attrs, down, up, categ_columns)
        if len(eq_class_synth) == 0:
            continue

        sensitive_columns = columns - attrs.keys()
        for sensitive_column in sensitive_columns:
            a = eq_class_orig[sensitive_column]
            b = eq_class_synth[sensitive_column]
            if emd_samples(a, b) < k_distance:
                result.append(attrs)
                break
    return result


def t_closeness_check(df, threshold=0.2):
    """
    Returns a list of dicts where each dict represents attributes that can be used
    to find equivalence classes which do not satisfy t-closeness requirement

    """

    def is_t_close(group, columns):
        for column in columns:
            a = df[column]
            b = group[column]
            if emd_samples(a, b) > threshold:
                return False
        return True

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
                result.extend(vulnerable_identifiers.to_dict('records'))
    return result


def find_neighbour_distances(df, attr_dict, categ_columns):
    up = {}
    down = {}
    for attr, val in attr_dict.items():
        if attr in categ_columns:
            continue
        higher = df[df[attr] > val][attr]
        lower = df[df[attr] < val][attr]
        if len(higher) > 0:
            up[attr] = higher.min() - val
        if len(lower) > 0:
            down[attr] = val - lower.max()
    return down, up


def find_eq_class(df, attrs):
    f = pd.Series([True] * len(df), index=df.index)
    for attr, val in attrs.items():
        f = f & (df[attr] == val)
    return df[f]


def find_eq_class_fuzzy(df, attrs, down, up, categ_columns):
    f = pd.Series([True] * len(df), index=df.index)
    for attr, val in attrs.items():
        if attr in categ_columns:
            f = f & (df[attr] == val)
        else:
            if attr in up:
                f = f & (df[attr] < val + up[attr] / 2.)
            if attr in down:
                f = f & (df[attr] > val - down[attr] / 2.)
    return df[f]
