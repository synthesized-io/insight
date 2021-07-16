import re
from typing import Dict

from ..common.rules import GenericRule, TableColumn


def replace_alias_with_table_name(
    query: str, alias_to_table_name: Dict[str, str], max_trials: int = 5
) -> str:
    """Find any ALIAS.COLUMN_NAME in a given string and replace it with TABLE_NAME.COLUMN_NAME.

    For example,
    >>> replace_alias_with_table_name(
            "CASE WHEN A.A = 'A' THEN B.B1 ELSE B.B2 END",
            {"A": "TABLE_A", "B": "TABLE_B": "B"}
        )
    >>> "CASE WHEN TABLE_A.A = 'A' THEN TABLE_B.B1 ELSE TABLE_B.B2 END"
    """
    query = " " + query.replace("\n", " ") + " "
    for alias_orig, table_name in alias_to_table_name.items():
        for alias in [alias_orig.lower(), alias_orig.upper()]:
            column_alias_regex = re.compile(rf"\W\[?{alias}\]?\.\[?\w*\]?\W")
            alias_regex = re.compile(rf"\[?{alias}\]?\.")

            def f_repl(matchobj):
                return re.sub(alias_regex, f"{table_name}.", matchobj.group(0))

            if alias == table_name:
                continue

            # In some cases where there is regex overlapping we need to subn twice.
            (query, _) = re.subn(column_alias_regex, f_repl, query)
            (query, _) = re.subn(column_alias_regex, f_repl, query)

    return query.strip()


def update_alias_table_name(rule: GenericRule, alias_to_table_name: Dict[str, str]):
    for children in rule.get_children():
        if isinstance(children, TableColumn):
            alias = children.table_name
            table_name = alias_to_table_name.get(alias, None) if alias else None
            if table_name:
                children.table_name = table_name
