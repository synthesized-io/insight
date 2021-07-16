import os
import uuid

import pytest

from synthesized.common.rules import (AllColumns, And, CaseWhen, Column, Equals, GenericRule, IsIn, IsNull, Length, Not,
                                      Sum, TableColumn, Value)
from synthesized.common.rules.function import Right
from synthesized.suite.parsers import ParserError, parse_join_stmt, parse_query, parse_rule_stmt
from synthesized.suite.parsers.join import ParsedJoinStmt
from synthesized.suite.parsers.query import Query
from synthesized.suite.parsers.rule import get_all_cols, get_null_cols, is_consistent_rule

from .cases import case_synthetic, case_tpch, rules_annotated, rules_raw


@pytest.mark.parametrize(
    "stmt,expected_tl,expected_raw,expected_attn",
    [
        (
            "FROM table_a a JOIN table_b b ON a.b_id = b.id",
            ["table_a", "table_b"],
            "FROM table_a JOIN table_b ON (table_a.b_id = table_b.id)",
            {
                "table_a": "table_a",
                "a": "table_a",
                "table_b": "table_b",
                "b": "table_b",
            },
        ),
        (
            "FROM table_a a JOIN table_b b ON a.b_id = b.id JOIN table_c c ON c.id = a.c_id",
            ["table_a", "table_b", "table_c"],
            "FROM table_a JOIN table_b ON (table_a.b_id = table_b.id) JOIN table_c ON (table_c.id = table_a.c_id)",
            {
                "table_a": "table_a",
                "a": "table_a",
                "table_b": "table_b",
                "b": "table_b",
                "table_c": "table_c",
                "c": "table_c",
            },
        ),
        (
            """from region r
                join nation n on r.r_regionkey = n.n_regionkey
                join supplier s on n.n_nationkey = s.s_nationkey
                join customer c on n.n_nationkey = c.c_nationkey
            """,
            ["region", "nation", "supplier", "customer"],
            (
                "FROM region"
                " JOIN nation ON (region.r_regionkey = nation.n_regionkey)"
                " JOIN supplier ON (nation.n_nationkey = supplier.s_nationkey)"
                " JOIN customer ON (nation.n_nationkey = customer.c_nationkey)"
            ),
            {
                "r": "region",
                "n": "nation",
                "s": "supplier",
                "c": "customer",
                "region": "region",
                "nation": "nation",
                "supplier": "supplier",
                "customer": "customer",
            },
        ),
        ("from table_a ABC", ["table_a"], "FROM table_a", {"abc": "table_a"}),
        ("FROM table_a", ["table_a"], "FROM table_a", {"table_a": "table_a"}),
        (
            "FROM table_a as a JOIN table_b as b ON a.b_id = b.id",
            ["table_a", "table_b"],
            "FROM table_a JOIN table_b ON (table_a.b_id = table_b.id)",
            {
                "table_a": "table_a",
                "a": "table_a",
                "table_b": "table_b",
                "b": "table_b",
            },
        ),
    ],
)
def test_suite_parsers_join(stmt, expected_tl, expected_raw, expected_attn):
    parsed = parse_join_stmt(stmt)
    assert sorted(parsed.table_list) == sorted(expected_tl)
    assert parsed.raw == expected_raw
    for key, value in expected_attn.items():
        # attn also contains identity
        assert parsed.alias_to_table_name[key] == value


rules_annotated_path = rules_annotated()


@pytest.mark.parametrize(
    "program_path,expected_length",
    [
        (os.path.join(rules_annotated_path, "0.txt"), 2),
        (os.path.join(rules_annotated_path, "1.txt"), 4),
        (os.path.join(rules_annotated_path, "2.txt"), 9),
        (os.path.join(rules_annotated_path, "3.txt"), 3),
        (os.path.join(rules_annotated_path, "4.txt"), 5),
        (os.path.join(rules_annotated_path, "5.txt"), 2),
        (os.path.join(rules_annotated_path, "6.txt"), 5),
    ],
)
def test_suite_parsers_rule_parse_strong(program_path, expected_length):
    with open(program_path) as f:
        unwinded_list = parse_rule_stmt(f.read())
        assert (
            len(unwinded_list) == expected_length
        ), f"len(unwinded_list)={len(unwinded_list)}, expected_length={expected_length}"

        for unwind_prop, unwind_value in unwinded_list:
            assert isinstance(unwind_prop, GenericRule)
            assert isinstance(unwind_value, GenericRule)


def test_suite_parsers_rule_parse_weak():
    some_columns = [
        "SAPBW_1_BPARTNER2",
        "CLOSE_1_ROLE_EXTERNALREFERENCE",
        "HOUSE_3_AANVRAGER_SILVERRECORDID",
        "DAYBREAK_WUBSAP_BPARTNER",
        "DAYBREAK_WUBSAP_BPARTNER",
        "DAYBREAK_WUB_CUSTOMER_NBR",
        "DAYBREAK_CUS_SSN",
        "HOUSE_3_AANVRAGER_SILVERRECORDID",
        "HOUSE_1_BACKOFFICEID",
        "HOUSE_2_MORTGAGEAPPLICATION_AANVRAAGID",
        "HOUSE_2_AANVRAGER_VOLGNR",
    ]

    for program in rules_raw():
        parse_rule_stmt(program, column_name_list=some_columns)


_, rule_stmt_df, _ = case_synthetic()


def test_suite_parsers_rule_parse_weak_synthetic():
    for _, (_, program) in rule_stmt_df.iterrows():
        parse_rule_stmt(program)


_, rule_stmt_df, _ = case_tpch()


def test_suite_parsers_rule_parse_weak_tpch():
    for _, (_, program) in rule_stmt_df.iterrows():
        parse_rule_stmt(program)


def test_suite_parsers_raises_ParserError():
    with pytest.raises(ParserError):
        parse_join_stmt(str(uuid.uuid4()))
    with pytest.raises(ParserError):
        parse_rule_stmt("CASE++")


@pytest.mark.parametrize(
    "stmt,expected_query",
    [
        (
            """
            select
                case when LEN(a.a) = 'hello' then 1 else 2 end,
                a.a + 2
            FROM table_a a JOIN table_b b ON a.b_id = b.id
            WHERE a.col1 = b.col1
            GROUP BY a.col2
            """,
            Query(
                rule_list=[
                    CaseWhen(
                        when=[
                            Equals(
                                Length([TableColumn(table_name="table_a", column_name="a")]),
                                Value(value="hello"),
                            ),
                            Not(
                                Equals(
                                    Length(
                                        [TableColumn(table_name="table_a", column_name="a")]
                                    ),
                                    Value(value="hello"),
                                )
                            ),
                        ],
                        then=[Value(value=1.0), Value(value=2.0)],
                        else_value=Value(value=None),
                    ),
                    Sum(
                        [
                            TableColumn(table_name="table_a", column_name="a"),
                            Value(value=2.0),
                        ]
                    ),
                ],
                join_stmt=ParsedJoinStmt(
                    table_list=["table_a", "table_b"],
                    alias_to_table_name={
                        "table_a": "table_a",
                        "a": "table_a",
                        "table_b": "table_b",
                        "b": "table_b",
                    },
                    main_table="table_a",
                    table_name_to_join_rule={
                        "table_b": Equals(
                            TableColumn(table_name="table_a", column_name="b_id"),
                            TableColumn(table_name="table_b", column_name="id"),
                        )
                    },
                    table_name_to_join_type={"table_b": None},
                    where_rule=Equals(
                        TableColumn(table_name="table_a", column_name="col1"),
                        TableColumn(table_name="table_b", column_name="col1"),
                    ),
                    groupby_rule=TableColumn(table_name="table_a", column_name="col2"),
                ),
            ),
        ),
        (
            """
            select
                a.a
            FROM table_a a
            GROUP BY a.col2""",
            Query(
                rule_list=[TableColumn(table_name="table_a", column_name="a")],
                join_stmt=ParsedJoinStmt(
                    table_list=["table_a"],
                    alias_to_table_name={"table_a": "table_a", "a": "table_a"},
                    main_table="table_a",
                    table_name_to_join_rule=None,
                    table_name_to_join_type=None,
                    where_rule=None,
                    groupby_rule=TableColumn(table_name="table_a", column_name="col2"),
                ),
            ),
        ),
        (
            """
            select
                a.a
            FROM table_a a
            WHERE a.col2 = 'hello'
            """,
            Query(
                rule_list=[TableColumn(table_name="table_a", column_name="a")],
                join_stmt=ParsedJoinStmt(
                    table_list=["table_a"],
                    alias_to_table_name={"table_a": "table_a", "a": "table_a"},
                    main_table="table_a",
                    table_name_to_join_rule=None,
                    table_name_to_join_type=None,
                    where_rule=Equals(
                        TableColumn(table_name="table_a", column_name="col2"),
                        Value(value="hello"),
                    ),
                    groupby_rule=None,
                ),
            ),
        ),
        (
            """
            select
                a.a
            FROM table_a a
            """,
            Query(
                rule_list=[TableColumn(table_name="table_a", column_name="a")],
                join_stmt=ParsedJoinStmt(
                    table_list=["table_a"],
                    alias_to_table_name={"table_a": "table_a", "a": "table_a"},
                    main_table="table_a",
                    table_name_to_join_rule=None,
                    table_name_to_join_type=None,
                ),
            ),
        ),
        (
            """
            select
                table_a.a
            FROM table_a
            """,
            Query(
                rule_list=[TableColumn(table_name="table_a", column_name="a")],
                join_stmt=ParsedJoinStmt(
                    table_list=["table_a"],
                    alias_to_table_name={"table_a": "table_a"},
                    main_table="table_a",
                    table_name_to_join_rule=None,
                    table_name_to_join_type=None,
                ),
            ),
        ),
        (
            """
            select *
            FROM table_a a
            """,
            Query(
                rule_list=[AllColumns()],
                join_stmt=ParsedJoinStmt(
                    table_list=["table_a"],
                    alias_to_table_name={"table_a": "table_a", "a": "table_a"},
                    main_table="table_a",
                    table_name_to_join_rule=None,
                    table_name_to_join_type=None,
                ),
            ),
        ),
    ],
)
def test_suite_parsers_query(stmt, expected_query):
    # raise ValueError
    parsed_query = parse_query(stmt)
    assert (
        parsed_query == expected_query
    ), f"parsed_query={parsed_query}\n\nexpected_query={expected_query}"




@pytest.mark.parametrize(
    "rule,null_cols,not_null_cols",
    [
        (IsNull(Column("A")), ["A"], []),
        (Not(IsNull(Column("A"))), [], ["A"]),
        (Equals(Column("A"), Value(1)), [], ["A"]),
        (Not(Equals(Column("A"), Value(1))), [], []),
        (Equals(Column("A"), Value(None)), ["A"], []),
        (Not(Equals(Column("A"), Value(None))), [], ["A"]),
        (IsIn(Column("A"), [Value(0), Value(1)]), [], ["A"]),
        (IsIn(Column("A"), [Value(0), Value(None)]), [], []),
        (Not(IsIn(Column("A"), [Value(0), Value(None)])), [], []),
        (IsIn(Column("A"), [Value(None), Value(None)]), ["A"], []),
        (Not(IsIn(Column("A"), [Value(None), Value(None)])), [], ["A"]),
    ]
)
def test_get_null_cols(rule, null_cols, not_null_cols):
    assert get_null_cols(rule) == (null_cols, not_null_cols)


@pytest.mark.parametrize(
    "rule,is_consistent",
    [
        (And([IsNull(Column("A")), Not(IsNull(Column("A")))]), False),
        (And([IsNull(Column("A")), Not(IsNull(Column("B")))]), True),
        (
            And([
                IsNull(TableColumn(table_name='invo_effectenrekening_referentie', column_name='ep_nn_securitiesaccount_accountnumber')),
                IsNull(TableColumn(table_name='invo_effectenrekening_referentie', column_name='ofs_investmentagreementaccount_accountnumber')),
                IsNull(TableColumn(table_name='invo_effectenrekening_referentie', column_name='ep_nn_securitiesaccount_accountnumber')),
                IsNull(TableColumn(table_name='invo_effectenrekening_referentie', column_name='ofs_investmentagreementaccount_accountnumber'))
            ]), True
        )
    ]
)
def test_is_consistent_rule(rule, is_consistent):
    assert is_consistent_rule(rule) == is_consistent


@pytest.mark.parametrize(
    "rule,cols",
    [
        (And([IsNull(TableColumn("A")), Not(IsNull(TableColumn("A")))]), ["A"]),
        (
            And([
                IsNull(TableColumn("A")),
                Right([TableColumn("B"), Value(3)]),
                Not(Equals(Length([TableColumn("C")]), Value(3)))
            ]), ["A", "B", "C"]),
    ]
)
def test_get_all_cols(rule, cols):
    assert sorted(get_all_cols(rule)) == sorted(cols)
