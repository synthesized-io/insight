from dataclasses import dataclass
from typing import List

from lark import Transformer
from lark.tree import Tree

from .grammar import load_grammar
from .join import ParsedJoinStmt, parse_join_stmt_from_tree
from .rule import parse_rule_stmt_from_tree
from ..helpers import update_alias_table_name
from ...common.rules import AllColumns, CaseWhen, GenericRule


@dataclass
class Query:
    rule_list: List[GenericRule]
    join_stmt: ParsedJoinStmt

    def __post_init__(self):
        for rule in self.rule_list:
            update_alias_table_name(rule, self.join_stmt.alias_to_table_name)

    def __repr__(self) -> str:
        rule_list_str = ",".join([f"\n\t\t{r}" for r in self.rule_list])
        return (
            "Query("
            f"\n\trule_list=[{rule_list_str}],"
            f"\n\tjoin_stmt={self.join_stmt}"
            "\n)"
        )


query_lr = load_grammar(start="query_specification")


class QueryTransformer(Transformer):
    def start(self, s):
        return s[0]

    def get_query(self, s):
        return Query(
            rule_list=s[-2],
            join_stmt=s[-1],
        )

    def table_expression(self, s):
        return parse_join_stmt_from_tree(Tree("start", s))

    def select_list(self, s):
        return s

    def get_query_expr(self, s) -> GenericRule:
        parsed_rules = parse_rule_stmt_from_tree(Tree("start", s))
        if len(parsed_rules) == 1 and parsed_rules[0][0] is None:
            return parsed_rules[0][1]

        return CaseWhen(
            when=[pred for pred, _ in parsed_rules],
            then=[value for _, value in parsed_rules],
        )

    def get_query_join_expr(self, s):
        (s,) = s
        return parse_join_stmt_from_tree(s)

    def all_columns(self, s):
        return AllColumns()


def parse_query(stmt: str) -> Query:
    uniform_stmt = stmt.replace("\n", " ").lower()
    parsed_tree = query_lr.parse(uniform_stmt)
    return QueryTransformer().transform(parsed_tree)
