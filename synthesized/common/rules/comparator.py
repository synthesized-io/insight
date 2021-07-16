from typing import Callable, Dict, List, cast

import numpy as np

from .base import GenericRule
from .function import Function
from .logic import And
from .node import Column, Value


class Comparator(GenericRule):
    pass


class ComparatorOneToOne(Comparator):
    def __init__(self, v1: GenericRule, v2: GenericRule) -> None:
        if not isinstance(v1, GenericRule):
            raise ValueError(
                f"Given values must be GenericRules, given v1={v1} has "
                f"type {v1.__class__.__name__}."
            )
        if not isinstance(v2, GenericRule):
            raise ValueError(
                f"Given values must be GenericRules, given v2={v2} has "
                f"type {v2.__class__.__name__}."
            )
        self.v1 = v1
        self.v2 = v2
        super().__init__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.v1}, {self.v2})"

    def get_children(self) -> List[GenericRule]:
        children = self.v1.get_children()
        children.extend(self.v2.get_children())
        return children


class ComparatorOneToMany(Comparator):
    def __init__(self, v1: GenericRule, v2: List[GenericRule]) -> None:
        if not isinstance(v1, GenericRule):
            raise ValueError(
                f"Given values must be GenericRules, given v1={v1} has "
                f"type {v1.__class__.__name__}."
            )
        if not isinstance(v2, list):
            raise ValueError(
                f"Given v2 must be List[GenericRules], given v2={v2} has "
                f"type {v2.__class__.__name__}."
            )
        if isinstance(v2, list) and any(
            [not isinstance(v2_i, GenericRule) for v2_i in v2]
        ):
            raise ValueError(
                f"Given values of v2 must be List[GenericRule], given v2={v2}"
            )

        self.v1 = v1
        self.v2 = v2
        super().__init__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.v1}, {self.v2})"

    def get_children(self) -> List[GenericRule]:
        children = self.v1.get_children()
        children.extend([c for v2_i in self.v2 for c in v2_i.get_children()])
        return children


class Equals(ComparatorOneToOne):
    def __init__(self, v1: GenericRule, v2: GenericRule) -> None:
        super().__init__(v1, v2)

    def to_pd_str(self, df_name: str) -> str:
        return f"({self.v1.to_pd_str(df_name=df_name)} == {self.v2.to_pd_str(df_name=df_name)})"

    def to_sql_str(self) -> str:
        return f"({self.v1.to_sql_str()} = {self.v2.to_sql_str()})"

    def get_augment_func(self, inverse: bool, **kwargs) -> Dict[str, Callable]:
        #  Case: When df[col_name] = 'value', e.g. args=[Column('AGE'), 32]
        if isinstance(self.v1, Column) and isinstance(self.v2, Value):
            return {self.v1.column_name: lambda _: "NA" if inverse else cast(Value, self.v2).value}

        #  Case: When 'value' = df[col_name], e.g. args=[32, Column('AGE')]
        if isinstance(self.v1, Value) and isinstance(self.v2, Column):
            return {self.v2.column_name: lambda _: "NA" if inverse else cast(Value, self.v1).value}

        #  Case: When df[col1] = df[col2], e.g. args=[Column('AGE1'), Column('AGE2')]
        if isinstance(self.v1, Column) and isinstance(self.v2, Column):
            return {
                self.v1.column_name: lambda row: (
                    row[cast(Column, self.v2).column_name] if not inverse
                    else add_random(row[cast(Column, self.v2).column_name])
                )
            }

        #  Case: When df[col_name].length = 'value', e.g. args=[('_func', ('LEN', [('_col_name', 'PASSWORD')])), 8]
        elif isinstance(self.v1, Function) and isinstance(self.v2, Value):
            return self.v1.get_augment_func(inverse=inverse, value=self.v2.value)

        return dict()


class GreaterThan(ComparatorOneToOne):
    def __init__(
        self, v1: GenericRule, v2: GenericRule, inclusive: bool = False
    ) -> None:
        super().__init__(v1, v2)
        self.inclusive = inclusive

    def to_pd_str(self, df_name: str) -> str:
        if self.inclusive:
            return f"({self.v1.to_pd_str(df_name=df_name)} >= {self.v2.to_pd_str(df_name=df_name)})"
        else:
            return f"({self.v1.to_pd_str(df_name=df_name)} > {self.v2.to_pd_str(df_name=df_name)})"

    def to_sql_str(self) -> str:
        if self.inclusive:
            return f"{self.v1.to_sql_str()} >= {self.v2.to_sql_str()}"
        else:
            return f"{self.v1.to_sql_str()} > {self.v2.to_sql_str()}"

    def get_augment_func(self, inverse: bool, **kwargs) -> Dict[str, Callable]:
        #  Case: When df[col_name] = 'value', e.g. args=[Column('AGE'), 32]
        if isinstance(self.v1, Column) and isinstance(self.v2, Value) and isinstance(self.v2.value, (int, float)):
            return {
                self.v1.column_name: lambda _: (
                    cast(Value, self.v2).value + add_random(cast(Value, self.v2).value) if not inverse
                    else cast(Value, self.v2).value - add_random(cast(Value, self.v2).value)
                )
            }

        #  Case: When 'value' = df[col_name], e.g. args=[32, Column('AGE')]
        if isinstance(self.v1, Value) and isinstance(self.v2, Column) and isinstance(self.v1.value, (int, float)):
            return {
                self.v2.column_name: lambda _: (
                    cast(Value, self.v1).value - add_random(cast(Value, self.v1).value) if not inverse
                    else cast(Value, self.v1).value + add_random(cast(Value, self.v1).value)
                )
            }

        #  Case: When df[col1] = df[col2], e.g. args=[Column('AGE1'), Column('AGE2')]
        if isinstance(self.v1, Column) and isinstance(self.v2, Column):
            return {
                cast(Column, self.v1).column_name: lambda row: (
                    row[cast(Column, self.v2).column_name] + add_random(row[cast(Column, self.v2).column_name])
                    if not inverse
                    else row[cast(Column, self.v2).column_name] - add_random(row[cast(Column, self.v2).column_name])
                )
            }

        #  Case: When df[col_name].length = 'value', e.g. args=[('_func', ('LEN', [('_col_name', 'PASSWORD')])), 8]
        elif isinstance(self.v1, Function) and isinstance(self.v2, Value):
            return self.v1.get_augment_func(inverse=inverse, value=self.v2.value)

        return dict()


class LowerThan(ComparatorOneToOne):
    def __init__(
        self, v1: GenericRule, v2: GenericRule, inclusive: bool = False
    ) -> None:
        super().__init__(v1, v2)
        self.inclusive = inclusive

    def to_pd_str(self, df_name: str) -> str:
        if self.inclusive:
            return f"({self.v1.to_pd_str(df_name=df_name)} <= {self.v2.to_pd_str(df_name=df_name)})"
        else:
            return f"({self.v1.to_pd_str(df_name=df_name)} < {self.v2.to_pd_str(df_name=df_name)})"

    def to_sql_str(self) -> str:
        if self.inclusive:
            return f"{self.v1.to_sql_str()} <= {self.v2.to_sql_str()}"
        else:
            return f"{self.v1.to_sql_str()} < {self.v2.to_sql_str()}"

    def get_augment_func(self, inverse: bool, **kwargs) -> Dict[str, Callable]:
        gt = GreaterThan(self.v2, self.v1, inclusive=not self.inclusive)
        return gt.get_augment_func(inverse=inverse, **kwargs)


class ValueRange(ComparatorOneToMany):
    def __init__(
        self,
        v1: GenericRule,
        v2: List[GenericRule],
        # In SQL, between operator is inclusive by default
        low_inclusive: bool = True,
        high_inclusive: bool = True,
    ) -> None:
        super().__init__(v1, v2)
        if len(v2) != 2:
            raise ValueError(f"ValueRange must be initialized with two values for v2, given {len(v2)}")

        self.low_inclusive = low_inclusive
        self.high_inclusive = high_inclusive

        self._rule = And([
            LowerThan(v2[0], v1, inclusive=low_inclusive),
            LowerThan(v1, v2[1], inclusive=high_inclusive)
        ])

    def to_pd_str(self, df_name: str) -> str:
        return self._rule.to_pd_str(df_name=df_name)

    def to_sql_str(self) -> str:
        if self.low_inclusive and self.high_inclusive:
            return f"{self.v1.to_sql_str()} BETWEEN {self.v2[0].to_sql_str()} AND {self.v2[1].to_sql_str()}"

        return self._rule.to_sql_str()

    def get_augment_func(self, inverse: bool, **kwargs) -> Dict[str, Callable]:
        #  Case: When df[col_name] = 'value', e.g. args=[Column('AGE'), 32]
        if (isinstance(self.v1, Column) and isinstance(self.v2[0], Value) and isinstance(self.v2[0].value, (int, float))
           and isinstance(self.v2[1], Value) and isinstance(self.v2[1].value, (int, float))):
            return {
                self.v1.column_name: lambda _: (
                    random_between(cast(Value, self.v2[0]).value, cast(Value, self.v2[1]).value) if not inverse
                    else cast(Value, self.v2[0]).value - add_random(cast(Value, self.v2[0]).value)
                )
            }
        return dict()


class IsIn(ComparatorOneToMany):
    def __init__(self, v1: GenericRule, v2: List[GenericRule]) -> None:
        super().__init__(v1, v2)

    def to_pd_str(self, df_name: str) -> str:
        pd_str_args = ", ".join([arg.to_pd_str(df_name=df_name) for arg in self.v2])
        return f"({self.v1.to_pd_str(df_name=df_name)}.apply(lambda x: x in [{pd_str_args}]))"

    def to_sql_str(self) -> str:
        sql_str_args = ", ".join([arg.to_sql_str() for arg in self.v2])
        return f"({self.v1.to_sql_str()} IN ({sql_str_args}))"

    def get_augment_func(self, inverse: bool, **kwargs) -> Dict[str, Callable]:
        #  Case: When df[col_name] = 'value', e.g. args=[Column('AGE'), 32]
        if isinstance(self.v1, Column) and any(
            [isinstance(v2i, Value) for v2i in self.v2]
        ):
            if not inverse:
                values = [v2i.value for v2i in self.v2 if isinstance(v2i, Value)]
                return {self.v1.column_name: lambda _: np.random.choice(values)}
            else:

                return {self.v1.column_name: lambda _: "NA"}

        return dict()


def add_random(x):
    return int(abs(x) * np.random.rand()) if x % 1 == 0 else abs(x) * np.random.rand()


def random_between(start, end):
    return np.random.randint(start, end) if start % 1 == 0 and end % 1 == 0 else np.random.uniform(start, end)
