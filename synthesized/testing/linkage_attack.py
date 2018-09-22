from itertools import combinations

import pandas as pd
import numpy as np
from pyemd import emd_samples

from .util import categorical_emd

NEAREST_NEIGHBOUR_MULT = 1.05
ENLARGED_NEIGHBOUR_MULT = 2.0
T_CLOSENESS_DEFAULT = 0.3
K_DISTANCE_DEFAULT = 0.02


class Column:
    def __init__(self, key_attribute, sensitive, categorical):
        self.key_attribute = key_attribute
        self.sensitive = sensitive
        self.categorical = categorical


def identify_attacks(df_orig, df_synth, schema, t_closeness=T_CLOSENESS_DEFAULT, k_distance=K_DISTANCE_DEFAULT):
    """
    Returns a dict of attacks with keys, values that correspond to the background knowledge of an attacker and lead to
    sensitive attribute disclosure

    """

    columns = set(df_orig.columns.values)
    result = []
    for attrs in t_closeness_check(df_orig, schema, t_closeness):
        down, up = find_neighbour_distances(df_orig, attrs, schema)

        eq_class_synth = find_eq_class_fuzzy(df_synth, attrs, down, up, schema)
        eq_class_orig = find_eq_class_fuzzy(df_orig, attrs, down, up, schema)
        if len(eq_class_synth) == 0:
            continue

        sensitive_columns = filter(lambda column: schema[column].sensitive, columns - attrs.keys())
        for sensitive_column in sensitive_columns:
            a = eq_class_orig[sensitive_column]
            b = eq_class_synth[sensitive_column]
            c = df_synth[sensitive_column]
            d = df_orig[sensitive_column]
            if schema[sensitive_column].categorical:
                emd_function = categorical_emd
            else:
                emd_function = emd_samples
            if emd_function(a, b) < k_distance and emd_function(b, c) > t_closeness and emd_function(a,
                                                                                                     d) > t_closeness:
                attack = {"knowledge": {k: {"value": v, "lower": down[k], "upper": up[k]} for k, v in attrs.items()},
                          "target": sensitive_column}
                result.append(attack)
                break
    return result


def get_df_subset(df, knowledge, schema):
    ind = pd.Series([False] * len(df), index=df.index)
    for k, v in knowledge.items():
        if schema[k].categorical:
            if isinstance(v["value"], list):
                for i in v["value"]:
                    ind = ind | (df[k] == i)
            else:
                ind = ind | (df[k] == v["value"])
        else:
            ind = ind | (df[k] <= v["value"] + v["upper"]) & (df[k] >= v["value"] - v["lower"])
    df = df[ind]
    return df


def show_attacked_data(df_orig, df_synth, attack, schema):
    orig_df_subset = get_df_subset(df_orig, attack["knowledge"], schema)
    synth_df_subset = get_df_subset(df_synth, attack["knowledge"], schema)
    print("attribute under attack: ", attack["target"])
    print("\nbacground knowledge: ", list(attack["knowledge"].keys())[0])
    print("\n\n original df subset: \n", orig_df_subset)
    print("\n\n synthetic df subset: \n", synth_df_subset)


def eradicate_attacks(df_orig, df_synth, attacks, schema, radical=False, t_closeness=T_CLOSENESS_DEFAULT,
                      k_distance=K_DISTANCE_DEFAULT):
    """
    Returns a dataframe cleared of all recurrent attacks

    """
    df = df_synth
    while len(attacks) != 0:
        print("remaining attacks : ", len(attacks))
        df = eradicate_attacks_iteration(df_orig, df, attacks, schema, radical, t_closeness, k_distance)
        attacks = identify_attacks(df_orig, df, schema)
    return df


def eradicate_attacks_iteration(df_orig, df_synth, attacks, schema, radical=False, t_closeness=T_CLOSENESS_DEFAULT,
                                k_distance=K_DISTANCE_DEFAULT):
    """
    Returns a dataframe cleared of current attacks

    """

    def enlarge_boundaries(df_synth, knowledge, schema):
        new_knowledge = {}
        for k, v in knowledge.items():
            if schema[k].categorical:
                # add randomly sampled category
                new_value = np.random.choice(df_synth[k], 1)[0]
                if isinstance(v["value"], list):
                    new_value = v["value"].append(new_value)
                else:
                    new_value = [v["value"], new_value]
                new_knowledge[k] = {"value": new_value, "lower": v["lower"], "upper": v["upper"] }
            else:
                new_knowledge[k] = {"value": v["value"], "lower": v["lower"] * ENLARGED_NEIGHBOUR_MULT,
                                    "upper": v["upper"] * ENLARGED_NEIGHBOUR_MULT}
        return new_knowledge

    def clear_df(df, attacks, schema):
        ind_final = pd.Series([False] * len(df), index=df.index)
        for attack in attacks:
            ind = pd.Series([True] * len(df), index=df.index)
            for k, v in attack["knowledge"].items():
                if schema[k].categorical:
                    ind = ind & (df[k] == v["value"])
                else:
                    ind = ind & ((df[k] <= v["value"] + v["upper"] * NEAREST_NEIGHBOUR_MULT) &
                                 (df[k] >= v["value"] - v["lower"] * NEAREST_NEIGHBOUR_MULT))
            ind_final = ind_final | ind
        df = df[~ind_final]
        return df

    cleared_df = clear_df(df_synth, attacks, schema)
    if radical == True or (np.abs(len(df_synth) - len(cleared_df)) / len(df_synth) < 0.02):
        return cleared_df
    for attack in attacks:
        target = attack["target"]
        knowledge = attack["knowledge"]
        eq_class_synth = get_df_subset(df_synth, knowledge, schema)
        enlarged_knowledge = enlarge_boundaries(df_synth, knowledge, schema)
        eq_class_synth_enlarged = get_df_subset(df_synth, enlarged_knowledge, schema)

        eq_class_orig = get_df_subset(df_orig, knowledge, schema)
        a = eq_class_orig[target]
        b = eq_class_synth[target]
        c = df_synth[target]
        d = df_orig[target]
        e = eq_class_synth_enlarged[target]
        if schema[target].categorical:
            emd_function = categorical_emd
        else:
            emd_function = emd_samples
        while emd_function(a, e) < k_distance and emd_function(e, c) > t_closeness and emd_function(a, d) > t_closeness:
            enlarged_knowledge = enlarge_boundaries(df_synth, enlarged_knowledge, schema)
            e = get_df_subset(df_synth, enlarge_boundaries(df_synth, enlarged_knowledge, schema), schema)[target]
        b = np.random.choice(e, len(b))
        while emd_function(a, b) < k_distance and emd_function(b, c) > t_closeness and emd_function(a, d) > t_closeness:
            b = np.random.choice(e, len(b))
        eq_class_synth[target] = b
        cleared_df = cleared_df.append(eq_class_synth, ignore_index=False)

    return cleared_df


def t_closeness_check(df, schema, threshold=0.2):
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
    """
    Returns two dicts with distances to the nearest neighbours for a given key attribute

    """
    up = {}
    down = {}
    for attr, val in attr_dict.items():
        if schema[attr].categorical:
            up[attr] = down[attr] = None
        higher = df[df[attr] > val][attr]
        lower = df[df[attr] < val][attr]
        if len(higher) > 0:
            up[attr] = higher.min() - val
        if len(lower) > 0:
            down[attr] = val - lower. max()
    return down, up


def find_eq_class(df, attrs):
    """
    Returns an equivalence class with exact matching of a key attribute

    """
    f = pd.Series([True] * len(df), index=df.index)
    for attr, val in attrs.items():
        f = f & (df[attr] == val)  # double comparison?
    return df[f]


def find_eq_class_fuzzy(df, attrs, down, up, schema):
    """
    Returns an equivalence class with fuzzy matching of a key attribute

    """
    ind = pd.Series([True] * len(df), index=df.index)
    for attr, val in attrs.items():
        if schema[attr].categorical:
            ind = ind & (df[attr] == val)
        else:
            if attr in up:
                ind = ind & (df[attr] < val + up[attr] * NEAREST_NEIGHBOUR_MULT)
            if attr in down:
                ind = ind & (df[attr] > val - down[attr] * NEAREST_NEIGHBOUR_MULT)
    return df[ind]
