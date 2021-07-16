from typing import List, Optional

from .base import GenericRule
from .node import Value


class CaseWhen(GenericRule):
    def __init__(
        self,
        when: List[GenericRule],
        then: List[GenericRule],
        else_value: Optional[GenericRule] = None,
    ) -> None:
        if not isinstance(when, list):
            raise ValueError(f"Argument when must be a list, given {type(when)}")
        elif any([not isinstance(i, GenericRule) for i in when]):
            raise ValueError("Argument when must be a list of GenericRules")
        if not isinstance(then, list):
            raise ValueError(f"Argument then must be a list, given {type(then)}")
        elif any([not isinstance(i, GenericRule) for i in then]):
            raise ValueError("Argument then must be a list of GenericRules")
        if else_value and not isinstance(else_value, GenericRule):
            raise ValueError(
                f"Argument else_value must be None or GenericRules, given {type(else_value)}"
            )

        self.when = when
        self.then = then
        self.else_value = else_value if else_value else Value(None)
        super().__init__()

    def __repr__(self) -> str:
        when_str = ", ".join([repr(w) for w in self.when])
        then_str = ", ".join([repr(t) for t in self.then])
        return f"CaseWhen(when=[{when_str}], then=[{then_str}], else_value={self.else_value})"

    def to_pd_str(self, df_name: str) -> str:
        raise NotImplementedError

    def to_sql_str(self) -> str:
        out_str = "CASE\n"
        for pred, val in zip(self.when, self.then):
            out_str += f"\tWHEN {pred.to_sql_str()} THEN {val.to_sql_str()}\n"
        out_str += "END"

        return out_str

    def get_children(self) -> List[GenericRule]:
        children = []
        for arg in self.when:
            children.extend(arg.get_children())
        for arg in self.then:
            children.extend(arg.get_children())
        children.append(self.else_value)
        return children
