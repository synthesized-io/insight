import logging
from abc import abstractmethod
from typing import Callable, Dict, List

import numpy as np

from .node import Column, GenericRule, UnhappyFlowNode

logger = logging.getLogger(__package__)


class LogicOperator(GenericRule):
    def __init__(self, argument: GenericRule) -> None:
        self.argument = argument
        super().__init__()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.argument})"

    def get_children(self) -> List["GenericRule"]:
        return self.argument.get_children()


class Not(LogicOperator):
    def __init__(self, argument: GenericRule) -> None:
        super().__init__(argument)

    def to_pd_str(self, df_name: str) -> str:
        return f"(~{self.argument.to_pd_str(df_name=df_name)})"

    def to_sql_str(self) -> str:
        return f"NOT ({self.argument.to_sql_str()})"

    def get_augment_func(self, inverse: bool, **kwargs) -> Dict[str, Callable]:
        return self.argument.get_augment_func(inverse=not inverse, **kwargs)


class IsNull(LogicOperator):
    def __init__(self, argument: GenericRule) -> None:
        super().__init__(argument)

    def to_pd_str(self, df_name: str) -> str:
        return f"({self.argument.to_pd_str(df_name=df_name)}.isna())"

    def to_sql_str(self) -> str:
        return f"{self.argument.to_sql_str()} IS NULL"

    def get_augment_func(self, inverse: bool, **kwargs) -> Dict[str, Callable]:
        for child in self.argument.get_children():
            if not isinstance(child, Column):
                logger.warning(f"Unable to generate null for {child}")
                continue

            col_name = child.column_name
            if not inverse:
                return {col_name: lambda _: np.nan}
            else:
                return {col_name: lambda _: np.random.randint(100)}

        return dict()


class Negate(LogicOperator):
    def __init__(self, argument: GenericRule) -> None:
        super().__init__(argument)

    def to_pd_str(self, df_name: str) -> str:
        return f"(-{self.argument.to_pd_str(df_name=df_name)})"

    def to_sql_str(self) -> str:
        return f"-{self.argument.to_sql_str()}"


class LogicComparator(GenericRule):
    def __init__(self, arguments: List[GenericRule]) -> None:
        if len(arguments) == 0:
            raise ValueError("Given empty list of arguments")
        self.arguments = arguments
        super().__init__()

    @staticmethod
    def clean_args(arguments):
        # Remove None/UnhappyFlowNodes
        arguments = [
            arg
            for arg in arguments
            if arg is not None and not isinstance(arg, UnhappyFlowNode)
        ]
        return arguments

    def __repr__(self):
        return f"{self.__class__.__name__}([{', '.join([repr(a) for a in self.arguments])}])"

    @abstractmethod
    def to_pd_str(self, df_name: str) -> str:
        raise NotImplementedError

    def get_augment_func(self, inverse: bool, **kwargs) -> Dict[str, Callable]:
        all_funcs = [arg.get_augment_func(inverse=inverse) for arg in self.arguments]
        affected_cols = [col for func in all_funcs for col in func]

        # If they all affect to independent columns
        funcs = {col: f_col for func in all_funcs for col, f_col in func.items()}
        if len(affected_cols) == len(set(affected_cols)):
            return funcs

        for arg in self.arguments:
            if isinstance(arg, Not) and isinstance(arg.argument, IsNull):
                arg_funcs = arg.get_augment_func(inverse=inverse)
                funcs.update(arg_funcs)

        for arg in self.arguments:
            if isinstance(arg, Not) and not isinstance(arg.argument, IsNull):
                arg_funcs = arg.get_augment_func(inverse=inverse)
                funcs.update(arg_funcs)

        # # If there are multiple functions that affect one column:
        for arg in self.arguments:
            if isinstance(arg, Not):
                continue

            arg_funcs = arg.get_augment_func(inverse=inverse)
            funcs.update(arg_funcs)

        return funcs

    def get_children(self) -> List[GenericRule]:
        children = []
        for arg in self.arguments:
            children.extend(arg.get_children())
        return children


class And(LogicComparator):
    def __init__(self, arguments=List[GenericRule]) -> None:
        arguments = self.clean_args(arguments)
        arguments = self.clean_and_args(arguments)
        super().__init__(arguments=arguments)

    def clean_and_args(self, arguments):
        # Find a And(And(A, B), C, And(D, E)) and unify it to And(A, B, C, D, E)
        if all([not isinstance(arg, And) for arg in arguments]):
            return arguments

        new_args = []
        for arg in arguments:
            if isinstance(arg, And):
                new_args.extend(arg.arguments)
            else:
                new_args.append(arg)

        return new_args

    def to_pd_str(self, df_name: str) -> str:
        return (
            "("
            + " & ".join(
                [arg_i.to_pd_str(df_name=df_name) for arg_i in self.arguments if arg_i]
            )
            + ")"
        )

    def to_sql_str(self) -> str:
        return (
            "("
            + " AND ".join([arg_i.to_sql_str() for arg_i in self.arguments if arg_i])
            + ")"
        )


class Or(LogicComparator):
    def __init__(self, arguments=List[GenericRule]) -> None:
        arguments = self.clean_args(arguments)
        arguments = self.clean_and_args(arguments)
        super().__init__(arguments=arguments)

    def clean_and_args(self, arguments):
        # Find a Or(Or(A, B), C, Or(D, E)) and unify it to Or(A, B, C, D, E)
        if all([not isinstance(arg, And) for arg in arguments]):
            return arguments

        new_args = []
        for arg in arguments:
            if isinstance(arg, Or):
                new_args.extend(arg.arguments)
            else:
                new_args.append(arg)

        return new_args

    def to_pd_str(self, df_name: str) -> str:
        return (
            "("
            + " | ".join(
                [arg_i.to_pd_str(df_name=df_name) for arg_i in self.arguments if arg_i]
            )
            + ")"
        )

    def to_sql_str(self) -> str:
        return (
            "("
            + " OR ".join([arg_i.to_sql_str() for arg_i in self.arguments if arg_i])
            + ")"
        )
