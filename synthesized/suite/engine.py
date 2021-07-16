import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .types import CoverageResult, ParsedRuleStmt, RuleStatement
from ..common.rules.base import GenericRule

logger = logging.getLogger(__package__)


def augment_df(df: pd.DataFrame, stmt: ParsedRuleStmt) -> pd.DataFrame:
    for pred, _ in stmt:
        if pred is None:
            continue

        funcs = pred.get_augment_func(inverse=False)

        df_idx = df.sample(1).copy()

        for col_name, func in funcs.items():
            if col_name not in df_idx.columns:
                new_col_name = guess_column_name_by_column_alias(col_name, df_idx)
                if new_col_name is None:
                    raise ValueError(f"Unable to find column {col_name}")
                col_name = new_col_name

            df_idx[col_name] = df_idx.apply(func, axis=1)
        df = df.append(df_idx).reset_index(drop=True)

    return df


def predicate_to_pandas(predicate: GenericRule, df_name: str) -> str:
    """
    Transforms a predicate into Python statement which filters the rows of df specified
        by `df_name` through the predicate.

    The result is such as if called like `df[<result>]` the resulting df will contain
        only the rows which match the predicate.

    NB:

        - the top-most string quotes are `'`, so the result could be safely eval'd
            in `"` quotes (like `eval(f"df[{result}]"))
        - the result is always enclosed in brackets (like
            `(df['table_a.col1']=='FINISHED')`)
    """
    return predicate.to_pd_str(df_name=df_name)


def predicate_to_sql(rules: List[Tuple[GenericRule, GenericRule]]) -> str:
    out_str = "CASE\n"
    for pred, val in rules:
        out_str += f"\tWHEN {pred.to_sql_str()} THEN {val.to_sql_str()}\n"
    out_str += "END"

    return out_str


def guess_column_name_by_column_alias(
    column_alias: str,
    df: pd.DataFrame,
) -> Optional[str]:
    try:
        return df.columns[
            [col.split(".")[-1].lower() for col in df.columns].index(
                column_alias.lower()
            )
        ]
    except ValueError:
        return None


def get_column_name_by_alias(
    aliased_name: str, df: pd.DataFrame, alias_to_table_name: Dict[str, str]
) -> Optional[str]:
    aliased_name = aliased_name.lower()
    if "." not in aliased_name:
        return guess_column_name_by_column_alias(aliased_name, df=df)

    splitted = aliased_name.split(".")
    assert (
        len(splitted) == 2
    ), f"Column name '{aliased_name}' can't contain multiple lookups"
    table_alias, column_alias = splitted
    assert (
        table_alias in alias_to_table_name
    ), f"Table alias {table_alias} not in lookup -- possibly wrong join? "
    table_name = alias_to_table_name[table_alias]
    column_name = column_alias.lower()
    return f"{table_name}.{column_name}"


def apply_predicate_to_df(
    predicate: GenericRule,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Applies predicate to a given df.
    """
    predicate_eval = predicate_to_pandas(predicate, df_name="df")
    logger.debug(f"evaluating `{predicate_eval}`")
    return eval(f"df[{predicate_eval}]", {"df": df})


def compute_data_coverage(
    df: pd.DataFrame,
    rule_list: List[RuleStatement],
    warning_not_covered: bool = False,
) -> CoverageResult:
    """
    Computes the how many rules in `rule_list` are covered by data in `df`.

    df is the result of join, meaning:

        - df.columns contain column names in format `table_name.column_name`
        - df.columns are all lowercase
    """
    n_predicates_covered = 0
    n_predicates_total = 0
    not_covered = []
    for rule in rule_list:
        stmt = rule.parsed_stmt
        n_predicates_total += len(stmt)
        for predicate, _ in stmt:
            if predicate is None and len(df) > 0:
                # If predicate is empty and the dataframe has at least 1 sample,
                #  we consider it is covered
                n_predicates_covered += 1
                continue

            df_applied = apply_predicate_to_df(predicate, df)
            if len(df_applied) > 0:
                n_predicates_covered += 1
                continue

            not_covered.append(predicate)
            if warning_not_covered:
                logger.warning(f"Rule not covered: {predicate}")

    coverage = n_predicates_covered * 1.0 / n_predicates_total if n_predicates_total > 0 else np.nan

    return CoverageResult(
        coverage=coverage,
        samples=len(df),
        not_covered_rules=[r.to_sql_str() for r in not_covered],
    )


def compute_rule_coverage(df: pd.DataFrame, rule: RuleStatement) -> float:
    stmt = rule.parsed_stmt
    if len(stmt) == 0:
        return 0.0
    n_predicates_covered = 0
    for predicate, _ in stmt:
        if predicate is None and len(df) > 0:
            n_predicates_covered += 1
            continue
        df_applied = apply_predicate_to_df(predicate, df)
        n_predicates_covered += int(len(df_applied) > 0)
    return n_predicates_covered * (1.0 / len(stmt))


def get_optimal_df(df: pd.DataFrame, rule_list: List[RuleStatement]) -> pd.DataFrame:
    all_indexes = set()
    for rule in rule_list:
        stmt = rule.parsed_stmt
        for predicate, _ in stmt:
            if predicate is None:
                continue
            rule_idxs = apply_predicate_to_df(predicate, df).index

            if all([idx not in all_indexes for idx in rule_idxs]) and len(rule_idxs) > 0:
                all_indexes.add(rule_idxs[0])

    return df[df.index.isin(all_indexes)]
