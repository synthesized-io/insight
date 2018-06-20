from itertools import combinations

import pandas as pd
from pyemd import emd_samples


class Column:
    def __init__(self, key_attribute, sensitive, categorical):
        self.key_attribute = key_attribute
        self.sensitive = sensitive
        self.categorical = categorical


def linkage_attack(df_orig, df_synth, schema, t_closeness=0.2, k_distance=0.8):
    """categorical sensitive columns are not supported yet"""
    columns = set(df_orig.columns.values)
    result = []
    for attrs in t_closeness_check(df_orig, schema, t_closeness):
        eq_class_orig = find_eq_class(df_orig, attrs)

        down, up = find_neighbour_distances(df_orig, attrs, schema)
        eq_class_synth = find_eq_class_fuzzy(df_synth, attrs, down, up, schema)
        if len(eq_class_synth) == 0:
            continue

        sensitive_columns = filter(lambda column: schema[column].sensitive, columns - attrs.keys())
        for sensitive_column in sensitive_columns:
            a = eq_class_orig[sensitive_column]
            b = eq_class_synth[sensitive_column]
            if emd_samples(a, b, bins=100) < k_distance:
                result.append(attrs)
                break
    return result


def t_closeness_check(df, schema, threshold=0.2):
    """
    Returns a list of dicts where each dict represents attributes that can be used
    to find equivalence classes which do not satisfy t-closeness requirement

    """

    def is_t_close(group, columns):
        for column in columns:
            a = df[column]
            b = group[column]
            if emd_samples(a, b, bins=100) > threshold:
                return False
        return True

    result = []
    columns = set(df.columns.values)
    all_key_columns = set(filter(lambda column: schema[column].key_attribute, columns))
    all_sensitive_columns = set(filter(lambda column: schema[column].sensitive, columns))
    for i in range(1, len(all_key_columns) + 1):
        for key_columns in combinations(all_key_columns, i):
            sensitive_columns = all_sensitive_columns - set(key_columns)
            if len(sensitive_columns) == 0:
                continue
            vulnerable_rows = df.groupby(by=list(key_columns)).filter(
                lambda g: not is_t_close(g, columns=sensitive_columns))
            key_attributes = vulnerable_rows[list(key_columns)].drop_duplicates()
            if not key_attributes.empty:
                result.extend(key_attributes.to_dict('records'))
    return result


def find_neighbour_distances(df, attr_dict, schema):
    up = {}
    down = {}
    for attr, val in attr_dict.items():
        if schema[attr].categorical:
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
        f = f & (df[attr] == val)  # double comparison?
    return df[f]


def find_eq_class_fuzzy(df, attrs, down, up, schema):
    f = pd.Series([True] * len(df), index=df.index)
    for attr, val in attrs.items():
        if schema[attr].categorical:
            f = f & (df[attr] == val)
        else:
            if attr in up:
                f = f & (df[attr] < val + up[attr] / 2.)
            if attr in down:
                f = f & (df[attr] > val - down[attr] / 2.)
    return df[f]

