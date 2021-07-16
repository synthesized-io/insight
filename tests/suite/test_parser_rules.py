import pandas as pd
import pytest

from synthesized.common.rules import (And, Equals, IsNull, Left, Length, Lower, Minus, Not, Right, Sum, TableColumn,
                                      UnhappyFlowNode, Upper, Value)
from synthesized.suite.engine import apply_predicate_to_df, predicate_to_pandas
from synthesized.suite.helpers import replace_alias_with_table_name
from synthesized.suite.parsers import parse_join_stmt, parse_rule_stmt

from .cases import case_synthetic


def test_data():
    df = pd.DataFrame(
        {
            "table_a.a": [943817, 1874, 101037, 321],
            "table_b.b": [94371, 48371, 101037, 321],
            "table_c.c": ["fdacda", "nmqjkx", 101037, 321],
            "table_d.d": ["fdacda", "nmqjkx", 101037, 321],
        }
    )
    alias_to_table_name = {
        "a": "table_a",
        "table_a": "table_a",
        "b": "table_b",
        "table_b": "table_b",
        "c": "table_c",
        "table_c": "table_c",
        "d": "table_d",
        "table_d": "table_d",
    }

    return df, alias_to_table_name


@pytest.mark.parametrize(
    "statement,expected_rules,expected_pd_stmts",
    [
        (
            "CASE WHEN LEN(a.a) = 1 THEN COALESCE(c.c, d.d, '??') ELSE 3 END",
            [
                (
                    And(
                        [
                            Equals(Length([TableColumn("a", "table_a")]), Value(value=1.0)),
                            Not(IsNull(TableColumn("c", "table_c"))),
                        ]
                    ),
                    TableColumn("c", "table_c"),
                ),
                (
                    And(
                        [
                            Equals(Length([TableColumn("a", "table_a")]), Value(value=1.0)),
                            IsNull(TableColumn("c", "table_c")),
                            Not(IsNull(TableColumn("d", "table_d"))),
                        ]
                    ),
                    TableColumn("d", "table_d"),
                ),
                (
                    And(
                        [
                            Equals(Length([TableColumn("a", "table_a")]), Value(value=1.0)),
                            IsNull(TableColumn("c", "table_c")),
                            IsNull(TableColumn("d", "table_d")),
                        ]
                    ),
                    UnhappyFlowNode(),
                ),
                (
                    Not(Equals(Length([TableColumn("a", "table_a")]), Value(value=1.0))),
                    Value(value=3.0),
                ),
            ],
            [
                "((df['table_a.a'].astype(str).str.len() == 1.0) & (~(df['table_c.c'].isna())))",
                "((df['table_a.a'].astype(str).str.len() == 1.0) & (df['table_c.c'].isna()) & (~(df['table_d.d'].isna())))",
                "((df['table_a.a'].astype(str).str.len() == 1.0) & (df['table_c.c'].isna()) & (df['table_d.d'].isna()))",
                "(~(df['table_a.a'].astype(str).str.len() == 1.0))",
            ],
        ),
        (
            "CASE WHEN a.a + b.b - 3 = 1 THEN LEN(COALESCE(c.c, d.d)) ELSE 3 END",
            [
                (
                    And(
                        [
                            Equals(
                                Sum(
                                    [
                                        TableColumn("a", "table_a"),
                                        Minus(
                                            [TableColumn("b", "table_b"), Value(value=3.0)]
                                        ),
                                    ]
                                ),
                                Value(value=1.0),
                            ),
                            Not(IsNull(TableColumn("c", "table_c"))),
                        ]
                    ),
                    Length([TableColumn("c", "table_c")]),
                ),
                (
                    And(
                        [
                            Equals(
                                Sum(
                                    [
                                        TableColumn("a", "table_a"),
                                        Minus(
                                            [TableColumn("b", "table_b"), Value(value=3.0)]
                                        ),
                                    ]
                                ),
                                Value(value=1.0),
                            ),
                            IsNull(TableColumn("c", "table_c")),
                            Not(IsNull(TableColumn("d", "table_d"))),
                        ]
                    ),
                    Length([TableColumn("d", "table_d")]),
                ),
                (
                    And(
                        [
                            Equals(
                                Sum(
                                    [
                                        TableColumn("a", "table_a"),
                                        Minus(
                                            [TableColumn("b", "table_b"), Value(value=3.0)]
                                        ),
                                    ]
                                ),
                                Value(value=1.0),
                            ),
                            IsNull(TableColumn("c", "table_c")),
                            IsNull(TableColumn("d", "table_d")),
                        ]
                    ),
                    UnhappyFlowNode(),
                ),
                (
                    Not(
                        Equals(
                            Sum(
                                [
                                    TableColumn("a", "table_a"),
                                    Minus([TableColumn("b", "table_b"), Value(value=3.0)]),
                                ]
                            ),
                            Value(value=1.0),
                        )
                    ),
                    Value(value=3.0),
                ),
            ],
            [
                "(((df['table_a.a'] + (df['table_b.b'] - 3.0)) == 1.0) & (~(df['table_c.c'].isna())))",
                "(((df['table_a.a'] + (df['table_b.b'] - 3.0)) == 1.0) & (df['table_c.c'].isna()) & (~(df['table_d.d'].isna())))",
                "(((df['table_a.a'] + (df['table_b.b'] - 3.0)) == 1.0) & (df['table_c.c'].isna()) & (df['table_d.d'].isna()))",
                "(~((df['table_a.a'] + (df['table_b.b'] - 3.0)) == 1.0))",
            ],
        ),
        (
            "CASE WHEN RIGHT(a.a, 2) = '0AC' THEN LEFT(b.b, 4) ELSE c.c END",
            [
                (
                    Equals(
                        Right([TableColumn("a", "table_a"), Value(value=2.0)]),
                        Value(value="0AC"),
                    ),
                    Left([TableColumn("b", "table_b"), Value(value=4.0)]),
                ),
                (
                    Not(
                        Equals(
                            Right([TableColumn("a", "table_a"), Value(value=2.0)]),
                            Value(value="0AC"),
                        )
                    ),
                    TableColumn("c", "table_c"),
                ),
            ],
            [
                "(df['table_a.a'].astype(str).str.slice(start=-2, stop=None) == '0AC')",
                "(~(df['table_a.a'].astype(str).str.slice(start=-2, stop=None) == '0AC'))",
            ],
        ),
        (
            "CASE WHEN LOWER(RIGHT(UPPER(a.a), 3)) = 'abc' THEN 0 ELSE 1 END",
            [
                (
                    Equals(
                        Lower(
                            [Right([Upper([TableColumn("a", "table_a")]), Value(value=3.0)])]
                        ),
                        Value(value="abc"),
                    ),
                    Value(value=0.0),
                ),
                (
                    Not(
                        Equals(
                            Lower(
                                [
                                    Right(
                                        [
                                            Upper([TableColumn("a", "table_a")]),
                                            Value(value=3.0),
                                        ]
                                    )
                                ]
                            ),
                            Value(value="abc"),
                        )
                    ),
                    Value(value=1.0),
                ),
            ],
            [
                "(df['table_a.a'].astype(str).str.upper().astype(str).str.slice(start=-3, stop=None).astype(str).str.lower() == 'abc')",
                "(~(df['table_a.a'].astype(str).str.upper().astype(str).str.slice(start=-3, stop=None).astype(str).str.lower() == 'abc'))",
            ],
        ),
        (
            "CASE WHEN LEN(A.A) = 1 THEN 0 WHEN B.B = 2 THEN COALESCE(C.C, D.D, 'NULL') ELSE 3 END",
            [
                (
                    Equals(Length([TableColumn("a", "table_a")]), Value(value=1.0)),
                    Value(value=0.0),
                ),
                (
                    And(
                        [
                            Equals(TableColumn("b", "table_b"), Value(value=2.0)),
                            Not(IsNull(TableColumn("c", "table_c"))),
                        ]
                    ),
                    TableColumn("c", "table_c"),
                ),
                (
                    And(
                        [
                            Equals(TableColumn("b", "table_b"), Value(value=2.0)),
                            IsNull(TableColumn("c", "table_c")),
                            Not(IsNull(TableColumn("d", "table_d"))),
                        ]
                    ),
                    TableColumn("d", "table_d"),
                ),
                (
                    And(
                        [
                            Equals(TableColumn("b", "table_b"), Value(value=2.0)),
                            IsNull(TableColumn("c", "table_c")),
                            IsNull(TableColumn("d", "table_d")),
                        ]
                    ),
                    UnhappyFlowNode(),
                ),
                (
                    And(
                        [
                            Not(
                                Equals(
                                    Length([TableColumn("a", "table_a")]), Value(value=1.0)
                                )
                            ),
                            Not(Equals(TableColumn("b", "table_b"), Value(value=2.0))),
                        ]
                    ),
                    Value(value=3.0),
                ),
            ],
            [
                "(df['table_a.a'].astype(str).str.len() == 1.0)",
                "((df['table_b.b'] == 2.0) & (~(df['table_c.c'].isna())))",
                "((df['table_b.b'] == 2.0) & (df['table_c.c'].isna()) & (~(df['table_d.d'].isna())))",
                "((df['table_b.b'] == 2.0) & (df['table_c.c'].isna()) & (df['table_d.d'].isna()))",
                "((~(df['table_a.a'].astype(str).str.len() == 1.0)) & (~(df['table_b.b'] == 2.0)))",
            ],
        ),
        (
            "CASE WHEN c.c = 'QUION' THEN ISNULL(a.a,'hola1') + '#' + ISNULL(b.b,'hola2') ELSE 1 END ",
            [
                (
                    And(
                        [
                            Equals(
                                TableColumn(table_name="table_c", column_name="c"),
                                Value(value="QUION"),
                            ),
                            Not(IsNull(TableColumn(table_name="table_a", column_name="a"))),
                            Not(IsNull(TableColumn(table_name="table_b", column_name="b"))),
                        ]
                    ),
                    Sum(
                        [
                            TableColumn(table_name="table_a", column_name="a"),
                            Sum(
                                [
                                    Value(value="#"),
                                    TableColumn(table_name="table_b", column_name="b"),
                                ]
                            ),
                        ]
                    ),
                ),
                (
                    And(
                        [
                            Equals(
                                TableColumn(table_name="table_c", column_name="c"),
                                Value(value="QUION"),
                            ),
                            Not(IsNull(TableColumn(table_name="table_a", column_name="a"))),
                            IsNull(TableColumn(table_name="table_b", column_name="b")),
                        ]
                    ),
                    Sum(
                        [
                            TableColumn(table_name="table_a", column_name="a"),
                            Sum([Value(value="#"), Value(value="hola2")]),
                        ]
                    ),
                ),
                (
                    And(
                        [
                            Equals(
                                TableColumn(table_name="table_c", column_name="c"),
                                Value(value="QUION"),
                            ),
                            IsNull(TableColumn(table_name="table_a", column_name="a")),
                            Not(IsNull(TableColumn(table_name="table_b", column_name="b"))),
                        ]
                    ),
                    Sum(
                        [
                            Value(value="hola1"),
                            Sum(
                                [
                                    Value(value="#"),
                                    TableColumn(table_name="table_b", column_name="b"),
                                ]
                            ),
                        ]
                    ),
                ),
                (
                    And(
                        [
                            Equals(
                                TableColumn(table_name="table_c", column_name="c"),
                                Value(value="QUION"),
                            ),
                            IsNull(TableColumn(table_name="table_a", column_name="a")),
                            IsNull(TableColumn(table_name="table_b", column_name="b")),
                        ]
                    ),
                    Sum(
                        [
                            Value(value="hola1"),
                            Sum([Value(value="#"), Value(value="hola2")]),
                        ]
                    ),
                ),
                (
                    Not(
                        Equals(
                            TableColumn(table_name="table_c", column_name="c"),
                            Value(value="QUION"),
                        )
                    ),
                    Value(value=1.0),
                ),
            ],
            [
                "((df['table_c.c'] == 'QUION') & (~(df['table_a.a'].isna())) & (~(df['table_b.b'].isna())))",
                "((df['table_c.c'] == 'QUION') & (~(df['table_a.a'].isna())) & (df['table_b.b'].isna()))",
                "((df['table_c.c'] == 'QUION') & (df['table_a.a'].isna()) & (~(df['table_b.b'].isna())))",
                "((df['table_c.c'] == 'QUION') & (df['table_a.a'].isna()) & (df['table_b.b'].isna()))",
                "(~(df['table_c.c'] == 'QUION'))",
            ],
        ),
    ],
)
def test_suite_parsers_statement_e2e(statement, expected_rules, expected_pd_stmts):
    df, alias_to_table_name = test_data()

    statement = replace_alias_with_table_name(statement, alias_to_table_name)

    rules = parse_rule_stmt(statement)

    for r, er in zip(rules, expected_rules):
        assert r == er, f"\n\n{statement}\n\n{r[0]}\n{er[0]}\n\n\n{r[1]}\n{er[1]}\n\n"

    pd_stmts = [predicate_to_pandas(pred, df_name="df") for pred, _ in expected_rules]

    assert pd_stmts == expected_pd_stmts

    for pd_stmt_pred in pd_stmts:
        eval(pd_stmt_pred, {"df": df})



selector, rule_stmt_df, join_stmt_df = case_synthetic()

target_to_join_stmt = {
    target: parse_join_stmt(join_stmt)
    for _, (target, join_stmt) in join_stmt_df.iterrows()
}
target_to_join_df = {
    target: selector.join(target_to_join_stmt[target])
    for _, (target, join_stmt) in join_stmt_df.iterrows()
}  # this takes some time (~20s)


@pytest.mark.parametrize(
    "target,rule_stmt",
    [(target, rule_stmt) for _, (target, rule_stmt) in rule_stmt_df.iterrows()],
)
def test_suite_engine_coverage_apply_predicate_weak(target, rule_stmt):
    parsed_join_stmt = target_to_join_stmt[target]
    df = target_to_join_df[target]

    rule_stmt = replace_alias_with_table_name(
        rule_stmt, parsed_join_stmt.alias_to_table_name
    )

    parsed_rule_stmt = parse_rule_stmt(rule_stmt)
    for predicate, _ in parsed_rule_stmt:
        apply_predicate_to_df(df=df, predicate=predicate)
