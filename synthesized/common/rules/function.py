import logging
import string
from typing import Callable, Dict, List, cast

import numpy as np

from .base import GenericRule
from .exceptions import GenerationError
from .node import Column, Value

logger = logging.getLogger(__name__)


class Function(GenericRule):
    def __init__(self, func_name: str, arguments: List[GenericRule]) -> None:
        self.validate_args(arguments)

        self.func_name = func_name
        self.arguments = arguments
        super().__init__()

    @staticmethod
    def from_name(func_name: str, arguments: List[GenericRule]) -> "Function":
        func_name = func_name.upper()
        if func_name == "LEN":
            return Length(arguments)
        elif func_name == "UPPER":
            return Upper(arguments)
        elif func_name == "LOWER":
            return Lower(arguments)
        elif func_name == "RIGHT":
            return Right(arguments)
        elif func_name == "LEFT":
            return Left(arguments)
        elif func_name == "SUBSTRING":
            return Substring(arguments)
        elif func_name == "CONCAT":
            return Concatenate(arguments)
        elif func_name == "DATEDIFF":
            return DateDifference(arguments)
        elif func_name == "IS_DATE":
            return IsDate(arguments)
        else:
            raise NotImplementedError(f"Given function '{func_name}' not implemented")

    def __repr__(self):
        return f"{self.__class__.__name__}([{', '.join([repr(a) for a in self.arguments])}])"

    def validate_args(self, arguments):
        if not isinstance(arguments, list):
            raise ValueError(f"Given arguments must be a list, given {type(arguments)}")

    def get_children(self) -> List[GenericRule]:
        children = []
        for arg in self.arguments:
            children.extend(arg.get_children())
        return children

    def to_sql_str(self) -> str:
        sql_str_args = ", ".join([arg.to_sql_str() for arg in self.arguments])
        return f"{self.func_name}({sql_str_args})"


class Length(Function):
    def __init__(self, arguments: List[GenericRule]) -> None:
        super().__init__("LEN", arguments)

    def to_pd_str(self, df_name: str) -> str:
        return f"{self.arguments[0].to_pd_str(df_name=df_name)}.astype(str).str.len()"

    def validate_args(self, arguments):
        super().validate_args(arguments)
        assert len(arguments) == 1

    def get_augment_func(self, inverse: bool, **kwargs) -> Dict[str, Callable]:
        value = kwargs.get("value", None)
        assert value, f"{self.func_name} needs a value to generate data"
        assert isinstance(self.arguments[0], Column)

        length = int(value)
        column_name = self.arguments[0].column_name
        if not inverse:
            random_str = _generate_random_str(length=length)
            return {column_name: lambda _: _generate_random_str(length=length)}
        else:
            new_length = length + np.random.randint(1, length)
            random_str = _generate_random_str(length=new_length)
            return {column_name: lambda _: random_str}


class Upper(Function):
    def __init__(self, arguments: List[GenericRule]) -> None:
        super().__init__("UPPER", arguments)

    def to_pd_str(self, df_name: str) -> str:
        return f"{self.arguments[0].to_pd_str(df_name=df_name)}.astype(str).str.upper()"

    def validate_args(self, arguments):
        super().validate_args(arguments)
        assert len(arguments) == 1

    def get_augment_func(self, inverse: bool, **kwargs) -> Dict[str, Callable]:
        value = str(kwargs.get("value", None))
        assert value, f"{self.func_name} needs a value to generate data"
        assert isinstance(self.arguments[0], Column)

        column_name = self.arguments[0].column_name
        if not inverse:
            if value != value.upper():
                raise GenerationError(
                    f"{self.func_name}({column_name}) will never be '{value}' as it's not uppercase"
                )
            return {column_name: lambda _: value}
        else:
            return {column_name: lambda _: _generate_random_str(length=len(value) + 1)}


class Lower(Function):
    def __init__(self, arguments: List[GenericRule]) -> None:
        super().__init__("LOWER", arguments)

    def to_pd_str(self, df_name: str) -> str:
        return f"{self.arguments[0].to_pd_str(df_name=df_name)}.astype(str).str.lower()"

    def validate_args(self, arguments):
        super().validate_args(arguments)
        assert len(arguments) == 1

    def get_augment_func(self, inverse: bool, **kwargs) -> Dict[str, Callable]:
        value = str(kwargs.get("value", None))
        assert value, f"{self.func_name} needs a value to generate data"
        assert isinstance(self.arguments[0], Column)

        column_name = self.arguments[0].column_name
        if not inverse:
            if value != value.lower():
                raise GenerationError(
                    f"{self.func_name}({column_name}) will never be '{value}' as it's not lowercase"
                )

            return {column_name: lambda _: value.lower()}
        else:
            return {column_name: lambda _: _generate_random_str(length=len(value) + 1)}


class Right(Function):
    def __init__(self, arguments: List[GenericRule]) -> None:
        super().__init__("RIGHT", arguments)

    def to_pd_str(self, df_name: str) -> str:
        arg = int(float(self.arguments[1].to_pd_str(df_name=df_name)))
        return f"{self.arguments[0].to_pd_str(df_name=df_name)}.astype(str).str.slice(start=-{arg}, stop=None)"

    def validate_args(self, arguments):
        super().validate_args(arguments)
        assert len(arguments) == 2
        assert isinstance(arguments[1], Value)

    def get_augment_func(self, inverse: bool, **kwargs) -> Dict[str, Callable]:
        expected_value = str(kwargs.get("value", None))
        assert expected_value, "RIGHT needs a value to generate data"
        if isinstance(self.arguments[0], Column):

            column_name = self.arguments[0].column_name
            assert (
                isinstance(self.arguments[1], Value)
                and self.arguments[1].value is not None
            )
            value_len = int(self.arguments[1].value)

            if not inverse:
                if len(expected_value) != value_len:
                    raise GenerationError(
                        f"{self.func_name}({column_name}, {value_len}) will never be '{expected_value}'"
                        " as lengths don't match."
                    )

                return {
                    column_name: lambda _: _generate_random_str(
                        length=np.random.randint(1, 5)
                    )
                    + expected_value,
                }
            else:
                return {
                    column_name: lambda _: _generate_random_str(
                        length=np.random.randint(1, 5) + value_len
                    )
                }

        elif isinstance(self.arguments[0], Sum):
            # Little Hack for NN. Sometimes they have RIGHT("000" + A.COL1, 3) == 'ABC',
            # and the Sum operation doesn't affect the RIGHT operaion.
            sum_args = self.arguments[0].arguments
            if len(sum_args) == 2 and isinstance(sum_args[0], Value):
                new_right = Right([sum_args[1], self.arguments[1]])
                return new_right.get_augment_func(inverse=inverse, **kwargs)

        return dict()


class Left(Function):
    def __init__(self, arguments: List[GenericRule]) -> None:
        super().__init__("LEFT", arguments)

    def to_pd_str(self, df_name: str) -> str:
        arg = int(float(self.arguments[1].to_pd_str(df_name=df_name)))
        return f"{self.arguments[0].to_pd_str(df_name=df_name)}.astype(str).str.slice(start=0, stop={arg})"

    def validate_args(self, arguments):
        super().validate_args(arguments)
        assert len(arguments) == 2

    def get_augment_func(self, inverse: bool, **kwargs) -> Dict[str, Callable]:
        expected_value = kwargs.get("value", None)
        assert expected_value, "LEFT needs a value to generate data"
        if not isinstance(self.arguments[0], Column) and isinstance(
            self.arguments[1], Value
        ):
            return dict()

        assert isinstance(self.arguments[0], Column)
        column_name = self.arguments[0].column_name
        assert (
            isinstance(self.arguments[1], Value) and self.arguments[1].value is not None
        )
        value_len = int(self.arguments[1].value)

        if not inverse:
            if len(expected_value) != value_len:
                raise GenerationError(
                    f"{self.func_name}({column_name}, {value_len}) will never be '{expected_value}'"
                    " as lengths don't match."
                )

            return {
                column_name: lambda _: expected_value
                + _generate_random_str(length=np.random.randint(1, 5)),
            }
        else:
            return {
                column_name: lambda _: _generate_random_str(
                    length=np.random.randint(1, 5) + value_len
                )
            }


class Substring(Function):
    def __init__(self, arguments: List[GenericRule]) -> None:
        super().__init__("SUBSTRING", arguments)

    def to_pd_str(self, df_name: str) -> str:
        arg = self.arguments[0].to_pd_str(df_name=df_name)
        start = int(float(self.arguments[1].to_pd_str(df_name=df_name)))
        end = int(float(self.arguments[2].to_pd_str(df_name=df_name)))
        return f"{arg}.astype(str).str.slice(start={start}, stop={end})"

    def validate_args(self, arguments):
        super().validate_args(arguments)
        assert len(arguments) == 3

    def get_augment_func(self, inverse: bool, **kwargs) -> Dict[str, Callable]:
        expected_value = kwargs.get("value", None)
        assert expected_value, "Right needs a value to generate data"
        assert isinstance(self.arguments[0], Column)
        assert (
            isinstance(self.arguments[1], Value) and self.arguments[1].value is not None
        )
        assert (
            isinstance(self.arguments[2], Value) and self.arguments[2].value is not None
        )

        column_name = self.arguments[0].column_name
        start = int(self.arguments[1].value)
        end = int(self.arguments[2].value)

        if not inverse:
            value_len = end - start
            if len(expected_value) != value_len:
                raise GenerationError(
                    f"{self.func_name}({column_name}, {start}, {end}) will never be "
                    f"'{expected_value}' as lengths don't match."
                )

            return {
                column_name: lambda _: _generate_random_str(length=start)
                + expected_value
                + _generate_random_str(length=np.random.randint(1, 5)),
            }
        else:
            value_len = start + end
            return {
                column_name: lambda _: _generate_random_str(
                    length=np.random.randint(1, 5) + value_len
                )
            }


class DateDifference(Function):
    def __init__(self, arguments: List[GenericRule]) -> None:
        super().__init__("DATEDIFF", arguments)

    def to_pd_str(self, df_name: str) -> str:
        interval = self._sql_interval_to_pd(cast(str, cast(Value, self.arguments[0]).value))
        start = self.arguments[1].to_pd_str(df_name=df_name)
        end = self.arguments[2].to_pd_str(df_name=df_name)
        return f"(pd.to_datetime({end}) - pd.to_datetime({start})).astype('timedelta64[{interval}]')"

    def validate_args(self, arguments):
        super().validate_args(arguments)
        assert len(arguments) == 3
        assert isinstance(arguments[0], Value) and isinstance(arguments[0].value, str)

    @staticmethod
    def _sql_interval_to_pd(sql_interval: str) -> str:
        sql_interval = sql_interval.lower()
        if sql_interval in ("year", "yyyy", "yy"):
            return "Y"
        # elif sql_interval in ("quarter", "qq", "q"):
        #     return "Quarter"
        elif sql_interval in ("month", "mm", "m"):
            return "M"
        # elif sql_interval in ("dayofyear"):
        #     return "Day of the year"
        elif sql_interval in ("day", "dy", "y"):
            return "D"
        elif sql_interval in ("week", "ww", "wk"):
            return "W"
        # elif sql_interval in ("weekday", "dw", "w"):
        #     return "Weekday"
        elif sql_interval in ("hour", "hh"):
            return "h"
        elif sql_interval in ("minute", "mi", "n"):
            return "m"
        elif sql_interval in ("second", "ss", "s"):
            return "s"
        elif sql_interval in ("millisecond", "ms"):
            return "ms"
        raise ValueError(f"Given sql_interval={sql_interval} not recognized")


class IsDate(Function):
    def __init__(self, arguments: List[GenericRule]) -> None:
        super().__init__("IS_DATE", arguments)

    def to_pd_str(self, df_name: str) -> str:
        return f"(~pd.to_datetime({self.arguments[0].to_pd_str(df_name=df_name)}, errors='coerce').isna())"

    def validate_args(self, arguments):
        super().validate_args(arguments)
        assert len(arguments) == 1


class Concatenate(Function):
    def __init__(self, arguments: List[GenericRule]) -> None:
        super().__init__("CONCAT", arguments)

    def to_pd_str(self, df_name: str) -> str:
        def get_pd_str(arg):
            return f"{arg.to_pd_str(df_name=df_name)}.astype(str)"

        return "(" + " + ".join([get_pd_str(arg) for arg in self.arguments]) + ")"

    def validate_args(self, arguments):
        super().validate_args(arguments)


class Operation(Function):
    @staticmethod
    def from_operator(
        operator: str, this: GenericRule, other: GenericRule
    ) -> "Operation":
        if operator == "_add":
            return Sum([this, other])
        elif operator == "_minus":
            return Minus([this, other])
        elif operator == "_prod":
            return Prod([this, other])
        elif operator == "_div":
            return Div([this, other])
        else:
            raise KeyError(f"Given operator '{operator}' not recognized")


class Sum(Operation):
    def __init__(self, arguments: List[GenericRule]) -> None:
        super().__init__("SUM", arguments)

    def to_pd_str(self, df_name: str) -> str:
        any_str = any(
            [isinstance(v, Value) and isinstance(v.value, str) for v in self.arguments]
        )
        if any_str:
            arg_pd = " + ".join(
                [
                    arg.to_pd_str(df_name=df_name) + ".astype(str)"
                    if isinstance(arg, Column)
                    else arg.to_pd_str(df_name=df_name)
                    for arg in self.arguments
                ]
            )
        else:
            arg_pd = " + ".join(
                [arg.to_pd_str(df_name=df_name) for arg in self.arguments]
            )

        return "(" + arg_pd + ")"


class Prod(Operation):
    def __init__(self, arguments: List[GenericRule]) -> None:
        super().__init__("PROD", arguments)

    def to_pd_str(self, df_name: str) -> str:
        return (
            "("
            + " * ".join([arg.to_pd_str(df_name=df_name) for arg in self.arguments])
            + ")"
        )


class Minus(Operation):
    def __init__(self, arguments: List[GenericRule]) -> None:
        super().__init__("MINUS", arguments)

    def validate_args(self, arguments):
        super().validate_args(arguments)
        assert len(arguments) == 2

    def to_pd_str(self, df_name: str) -> str:
        return f"({self.arguments[0].to_pd_str(df_name=df_name)} - {self.arguments[1].to_pd_str(df_name=df_name)})"


class Div(Operation):
    def __init__(self, arguments: List[GenericRule]) -> None:
        super().__init__("DIV", arguments)

    def validate_args(self, arguments):
        super().validate_args(arguments)
        assert len(arguments) == 2

    def to_pd_str(self, df_name: str) -> str:
        return f"({self.arguments[0].to_pd_str(df_name=df_name)} / {self.arguments[1].to_pd_str(df_name=df_name)})"


def _generate_random_str(length: int) -> str:
    values = [s for s in string.hexdigits]
    value = "".join(np.random.choice(values) for _ in range(length))
    return value
