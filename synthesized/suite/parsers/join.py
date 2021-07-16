from dataclasses import dataclass
from typing import Dict, List, Optional

from lark import Transformer, Tree

from .grammar import load_grammar
from .rule import RuleTransformer, TransformedExpr, TransformedPredicate
from ..helpers import replace_alias_with_table_name, update_alias_table_name
from ...common.rules import CaseWhen, GenericRule


@dataclass
class ParsedJoinStmt:
    table_list: List[str]
    alias_to_table_name: Dict[str, str]
    main_table: str

    table_name_to_join_rule: Optional[Dict[str, GenericRule]] = None
    table_name_to_join_type: Optional[Dict[str, Optional[str]]] = None
    where_rule: Optional[GenericRule] = None
    groupby_rule: Optional[GenericRule] = None

    def __eq__(self, o: "ParsedJoinStmt") -> bool:  # type: ignore
        if sorted(self.table_list) != sorted(o.table_list):
            return False
        if self.alias_to_table_name != o.alias_to_table_name:
            return False
        if self.table_name_to_join_rule != o.table_name_to_join_rule:
            return False
        if self.table_name_to_join_type != o.table_name_to_join_type:
            return False
        if self.main_table != o.main_table:
            return False
        if self.where_rule != o.where_rule:
            return False
        if self.groupby_rule != o.groupby_rule:
            return False

        return True

    def __post_init__(self):
        # Make sure all tables are refered by name, not alias
        self.main_table = self.alias_to_table_name[self.main_table.lower()]
        if self.table_name_to_join_rule:
            new_table_name_to_join_rule = dict()
            for table_name, join_rule in self.table_name_to_join_rule.items():
                update_alias_table_name(join_rule, self.alias_to_table_name)
                new_table_name_to_join_rule[table_name] = join_rule
            self.table_name_to_join_rule = new_table_name_to_join_rule
        if self.where_rule:
            update_alias_table_name(self.where_rule, self.alias_to_table_name)
        if self.groupby_rule:
            update_alias_table_name(self.groupby_rule, self.alias_to_table_name)
        if self.raw:
            replace_alias_with_table_name(self.raw, self.alias_to_table_name)

    @property
    def raw(self):
        out = f"FROM {self.main_table}"

        if self.table_name_to_join_rule:
            for joined_table, join_rule in self.table_name_to_join_rule.items():
                if self.table_name_to_join_type[joined_table]:
                    out += f" {self.table_name_to_join_type[joined_table]}"
                out += (
                    f" JOIN {self.alias_to_table_name[joined_table.lower()]}"
                    f" ON {join_rule.to_sql_str()}"
                )

        if self.where_rule:
            out += f" WHERE {self.where_rule.to_sql_str()}"
        if self.groupby_rule:
            out += f" GROUP BY {self.groupby_rule.to_sql_str()}"

        return out


join_lr = load_grammar(start="table_expression")


class JoinTransformer(Transformer):
    alias_to_table_name: Dict[str, str] = {}
    table_to_schema: Dict[str, str] = {}
    main_table: Optional[str] = None
    table_name_to_join_rule: Dict[str, GenericRule] = {}
    table_name_to_join_type: Dict[str, str] = {}
    where_rule: Optional[GenericRule] = None
    groupby_rule: Optional[GenericRule] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alias_to_table_name = dict()
        self.table_to_schema = dict()
        self.table_name_to_join_rule = dict()
        self.table_name_to_join_type = dict()

    def update_rules_alias(self):
        if self.where_rule:
            update_alias_table_name(self.where_rule, self.alias_to_table_name)
        if self.groupby_rule:
            update_alias_table_name(self.groupby_rule, self.alias_to_table_name)

    def transform(self, tree: Tree):
        transformed_tree = super().transform(tree)
        self.update_rules_alias()
        return transformed_tree

    def table_expression(self, s):
        return s[0]

    def join_expr(self, s):
        self.main_table = s[0]
        return s

    def start(self, s):
        return s[0]

    def varname(self, s):
        return str(s[0])

    def join_rule(self, s):
        (s,) = s
        pred = RuleTransformer().transform(s)
        assert isinstance(pred, TransformedPredicate)
        parsed_rules = pred.pred
        return parsed_rules

    def join_type(self, s):
        (s,) = s
        return s

    def join_item_expr(self, s):
        if len(s) == 3:
            join_type, alias, join_rule = s
        else:
            join_type = None
            alias, join_rule = s

        table_name = self.alias_to_table_name[alias.lower()]

        self.table_name_to_join_rule[table_name] = join_rule
        self.table_name_to_join_type[table_name] = join_type
        return alias, join_rule

    def table_with_alias(self, s):
        # table name is an alias to itself
        self.alias_to_table_name[s[0].lower()] = s[0].lower()
        if len(s) > 1:
            # also add alias
            self.alias_to_table_name[s[1].lower()] = s[0].lower()
        return s[-1]  # return lookup-able alias

    def table_with_schema(self, s):
        """Store the table_to_schema and return table without schema"""
        if len(s) == 2:
            self.table_to_schema[s[1]] = s[0]
        return s[-1]

    def column_lookup(self, s):
        table_alias, column_name = s
        assert table_alias in self.alias_to_table_name
        return f"{self.alias_to_table_name[table_alias]}.{column_name}"

    def get_where_expr(self, s):
        (s,) = s
        pred = RuleTransformer().transform(s)
        assert isinstance(pred, TransformedPredicate)
        self.where_rule = pred.pred

    def get_groupby_expr(self, s):
        (s,) = s
        rule = RuleTransformer().transform(s)
        assert isinstance(rule, TransformedExpr)
        if rule.is_pred_empty:
            value = rule.to_value()
        else:
            value = CaseWhen(when=rule.pred_list, then=rule.value_list)

        self.groupby_rule = value


class JoinStatementRawTransformer(JoinTransformer):
    """
    Transforms the parsed tree into the SQL statement understandable by both
        PostgresSQL and PasdasQL.
    """

    def table_expression(self, s):
        join_expr = super().table_expression(s)

        out = join_expr
        if self.where_rule:
            out += f" WHERE {self.where_rule.to_sql_str()}"
        if self.groupby_rule:
            out += f" GROUP BY {self.groupby_rule.to_sql_str()}"

        return out

    def join_expr(self, s):
        s = super().join_expr(s)
        table_name = s[0]
        joined_list = s[1:]

        out = f"FROM {self.alias_to_table_name[table_name.lower()]}"
        for joined_alias, join_rule in joined_list:
            joined_table = self.alias_to_table_name[joined_alias.lower()]

            if self.table_name_to_join_type[joined_table]:
                out += f" {self.table_name_to_join_type[joined_table]}"
            out += (
                f" JOIN {self.alias_to_table_name[joined_table.lower()]}"
                f" ON {join_rule.to_sql_str()}"
            )

        return out.strip()


def parse_join_stmt(stmt: str) -> ParsedJoinStmt:
    """
    Extracts from join statement affected table list and PSQL dialect statement.

    All table aliases will be converted to lower case.

    Examples of input join statement:

    ```
    FROM table_a a JOIN table_b b ON a.b_id = b.id
    FROM table_a JOIN table_b ON table_a.b_id = table_b.id
    from table_a
    ```

    Sample result (1-2):

    ```
    {
        "table_list": ["table_a", "table_b"],
        "raw": "FROM table_a JOIN table_b ON table_a.b_id = table_b.id",
        "alias_to_table_name": {
            "a": "table_a",
            "b": "table_b",
            "table_a": "table_a",
            "table_b": "table_b"
        },
        "main_table": "table_a"
    }
    ```

    Sample result (3):

    ```
    {
        "table_list": ["table_a"],
        "raw": "FROM table_a",
        "alias_to_table_name": {"table_a": "table_a"},
        "main_table": "table_a"
    }
    ```
    """
    uniform_stmt = stmt.replace("\n", " ")
    parsed_tree = join_lr.parse(uniform_stmt)
    return parse_join_stmt_from_tree(parsed_tree)


def parse_join_stmt_from_tree(parsed_tree):
    transformer = JoinStatementRawTransformer()
    transformer.transform(parsed_tree)

    table_name_to_join_rule = (
        transformer.table_name_to_join_rule
        if transformer.table_name_to_join_rule
        else None
    )
    table_name_to_join_type = (
        transformer.table_name_to_join_type
        if transformer.table_name_to_join_type
        else None
    )

    return ParsedJoinStmt(
        table_list=list(set(transformer.alias_to_table_name.values())),
        alias_to_table_name=transformer.alias_to_table_name,
        main_table=transformer.main_table,
        table_name_to_join_rule=table_name_to_join_rule,
        table_name_to_join_type=table_name_to_join_type,
        where_rule=transformer.where_rule,
        groupby_rule=transformer.groupby_rule,
    )
