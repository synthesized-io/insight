from typing import List, Optional, Union

from .base import GenericRule


class RuleNode(GenericRule):
    def __init__(self) -> None:
        super().__init__()

    def get_children(self) -> List[GenericRule]:
        return [self]


class Column(RuleNode):
    def __init__(self, column_name: str) -> None:
        self.column_name = column_name
        super().__init__()

    def __repr__(self) -> str:
        return f"Column(column_name='{self.column_name}')"

    def to_pd_str(self, df_name: str) -> str:
        return f"{df_name}['{self.column_name}']"

    def to_sql_str(self) -> str:
        return f"{self.column_name}"


class TableColumn(Column):
    """Special type of columns, used only for databases"""
    def __init__(self, column_name: str, table_name: Optional[str] = None) -> None:
        self._column_name = column_name
        self.table_name = table_name

    @property
    def column_name(self):
        if self.table_name:
            return f"{self.table_name}.{self._column_name}"
        else:
            return self._column_name

    def __repr__(self) -> str:
        if self.table_name is None:
            return f"TableColumn(column_name='{self._column_name}')"
        return (
            f"TableColumn(table_name='{self.table_name}', column_name='{self._column_name}')"
        )

    def to_pd_str(self, df_name: str) -> str:
        if self.table_name:
            return f"{df_name}['{self.column_name}']"

        return f"{df_name}[[c for c in {df_name}.columns if c.endswith('{self._column_name}')][0]]"

    def to_sql_str(self) -> str:
        return self.column_name


class Value(RuleNode):
    def __init__(self, value: Union[float, int, str, None]) -> None:
        if value is not None and not isinstance(value, (float, int, str)):
            raise ValueError(f"Wrong value type, given {value} type {type(value)}")
        self.value = value
        super().__init__()

    def __repr__(self) -> str:
        if isinstance(self.value, str):
            return f"Value(value='{self.value}')"
        return f"Value(value={self.value})"

    def to_pd_str(self, df_name: str) -> str:
        return self._to_str()

    def to_sql_str(self) -> str:
        return self._to_str()

    def _to_str(self) -> str:
        if isinstance(self.value, str):
            return f"'{self.value}'"
        return f"{self.value}"


class UnhappyFlowNode(Value):
    def __init__(self) -> None:
        super().__init__("N/A")

    def __repr__(self) -> str:
        return "UnhappyFlowNode()"

    def to_pd_str(self, df_name: str) -> str:
        return "'N/A'"


class AllColumns(RuleNode):
    def __init__(self, table_name: Optional[str] = None) -> None:
        self.table_name = table_name
        super().__init__()

    def __repr__(self) -> str:
        if self.table_name:
            return f"AllColumns(table_name='{self.table_name}')"
        return "AllColumns()"

    def to_pd_str(self, df_name: str) -> str:
        return f"{df_name}[[c for c in {df_name}.columns if c.startswith('{self.table_name}')]]"

    def to_sql_str(self) -> str:
        if self.table_name:
            return f"[{self.table_name}].*)"
        return "*"
