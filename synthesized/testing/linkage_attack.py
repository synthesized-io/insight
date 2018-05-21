from itertools import combinations

from pyemd import emd_samples


def linkage_attack(df_orig, df_synth, categ_columns, t_closeness=0.2):
    """categorical columns are not supported yet"""
    result = []
    for attrs in t_closeness_check(df_orig, t_closeness):
        eq_class_orig = find_eq_class(df_orig, attrs)

        down, up = find_neighbour_distances(df_orig, attrs, categ_columns)
        eq_class_synth = find_eq_class_fuzzy(df_synth, attrs, down, up, categ_columns)

        for attr, _ in attrs.items():
            a = eq_class_orig[attr]
            b = eq_class_synth[attr]
            if emd_samples(a, b) < t_closeness:
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
        higher = df[df[attr] > val][attr].sort_values()
        lower = df[df[attr] < val][attr].sort_values(ascending=False)
        if len(higher) > 0:
            up[attr] = higher.iloc[0] - val
        if len(lower) > 0:
            down[attr] = val - lower.iloc[0]
    return down, up


def find_eq_class(df, attrs):
    f = None
    for attr, val in attrs.items():
        f = _add_filter(f, df[attr] == val)
    return df[f]


def find_eq_class_fuzzy(df, attrs, down, up, categ_columns):
    f = None
    for attr, val in attrs.items():
        if attr in categ_columns:
            f = _add_filter(f, df[attr] == val)
        else:
            if attr in up:
                f = _add_filter(f, df[attr] < val + up[attr] / 2.)
            if attr in down:
                f = _add_filter(f, df[attr] > val - down[attr] / 2.)
    return df[f]


def _add_filter(f, filter):
    if f is None:
        return filter
    else:
        return f & filter