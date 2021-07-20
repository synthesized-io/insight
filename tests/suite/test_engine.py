import pandas as pd
import pytest

from synthesized.common.rules import (And, Column, Equals, IsIn, IsNull, Left, Length, Not, Or, Sum, TableColumn, Value,
                                      ValueRange)
from synthesized.common.rules.function import Right, Substring
from synthesized.suite.engine import augment_df, compute_rule_coverage, get_optimal_df, predicate_to_pandas
from synthesized.suite.helpers import replace_alias_with_table_name
from synthesized.suite.types import RuleStatement


@pytest.mark.parametrize(
    "predicate,expected",
    [
        (
            Equals(TableColumn("col1", "table_a"), Value("FINISHED")),
            "(df['table_a.col1'] == 'FINISHED')",
        ),
        (
            And(
                [
                    Not(Equals(TableColumn("col1", "table_a"), Value("FINISHED"))),
                    Equals(TableColumn("col1", "table_b"), Value("FAILED")),
                ]
            ),
            "((~(df['table_a.col1'] == 'FINISHED')) & (df['table_b.col1'] == 'FAILED'))",
        ),
        (
            Or(
                [
                    Not(Equals(TableColumn("col1", "table_a"), Value("FINISHED"))),
                    Equals(TableColumn("col1", "table_b"), Value("FAILED")),
                ]
            ),
            "((~(df['table_a.col1'] == 'FINISHED')) | (df['table_b.col1'] == 'FAILED'))",
        ),
        (
            IsIn(TableColumn("col1", "table_a"), [Value("HOMES"), Value("MIDAS")]),
            "(df['table_a.col1'].apply(lambda x: x in ['HOMES', 'MIDAS']))",
        ),
        (
            IsNull(Length([TableColumn("col1", "table_a")])),
            "(df['table_a.col1'].astype(str).str.len().isna())",
        ),
        (
            Sum([Length([TableColumn("col1", "table_a")]), TableColumn("col2", "table_a"), Value("CONSTNAME")]),
            "(df['table_a.col1'].astype(str).str.len() + df['table_a.col2'].astype(str) + 'CONSTNAME')",
        ),
    ],
)
def test_suite_engine_coverage_predicate2pandas(predicate, expected):
    pandas = predicate_to_pandas(predicate, df_name="df")
    assert pandas.replace(" ", "") == expected.replace(" ", "")


@pytest.mark.parametrize(
    "in_str,expected_out_str",
    [
        (
            "CASE WHEN A.A1 = 'A1' AND A.A2 = 'A2' THEN B.B1 ELSE B.B2 END",
            "CASE WHEN table_a.A1 = 'A1' AND table_a.A2 = 'A2' THEN table_b.B1 ELSE table_b.B2 END",
        ),
        (
            "CASE WHEN LEN(A.A1) = 'A1' AND CAST(A.A2 as VARCHAR) = 'A2' THEN B.B1 ELSE B.B2 END",
            "CASE WHEN LEN(table_a.A1) = 'A1' AND CAST(table_a.A2 as VARCHAR) = 'A2' THEN table_b.B1 ELSE table_b.B2 END",
        ),
        ("A.column1", "table_a.column1"),
        (
            "A.COLUMN1+A.COLUMN2+A.COLUMN3+A.COLUMN4+A.COLUMN5",
            "table_a.COLUMN1+table_a.COLUMN2+table_a.COLUMN3+table_a.COLUMN4+table_a.COLUMN5",
        ),
    ],
)
def test_suite_replace_alias(in_str, expected_out_str):
    alias_to_table_name = {
        "table_a": "table_a",
        "a": "table_a",
        "table_b": "table_b",
        "b": "table_b",
    }
    out_str = replace_alias_with_table_name(in_str, alias_to_table_name)

    assert out_str == expected_out_str


@pytest.mark.parametrize(
    "rule,expected_coverage",
    [
        (ValueRange(Column('x1'), [Value(2), Value(4)]), 1),
        (ValueRange(Column('x1'), [Value(8), Value(10)]), 0),
        (Equals(Sum([Column('x1'), Column('x2')]), Value(5)), 1),
        (IsIn(Column('y'), [Value('A'), Value('B')]), 1),
        (IsIn(Column('y'), [Value('Y'), Value('Z')]), 0),
        (Equals(Column('y'), Value('A')), 1),
        (Equals(Column('y'), Value('Z')), 0),
        (Equals(Length([Column('y')]), Value(3)), 0),
        (Not(Equals(Length([Column('y')]), Value(3))), 1),
        (Equals(Right([Column('y'), Value(2)]), Value('AA')), 0),
        (Not(Equals(Right([Column('y'), Value(2)]), Value('AA'))), 1),
        (Equals(Left([Column('y'), Value(2)]), Value('AA')), 0),
        (Equals(Substring([Column('y'), Value(2), Value(4)]), Value('AA')), 0),
    ]
)
def test_coverage(rule, expected_coverage):
    df = pd.DataFrame(dict(
        x1=[0, 1, 2, 3, 4, 5],
        x2=[5, 4, 3, 2, 1, 0],
        y=['A', 'A', 'B', 'B', 'C', 'C'],
    ))
    parsed_rule_stmt = [(rule, "")]
    rule_stmt = RuleStatement("target", "", parsed_rule_stmt)

    coverage = compute_rule_coverage(df, rule_stmt)
    assert coverage == expected_coverage

    df_new = augment_df(df, parsed_rule_stmt)
    df_new = get_optimal_df(df_new, [rule_stmt])
    assert len(df_new) == 1

    coverage_new = compute_rule_coverage(df_new, rule_stmt)
    assert coverage_new == 1
