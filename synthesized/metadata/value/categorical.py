from typing import Dict, Optional, Sequence, cast

from ..base import Nominal


class String(Nominal[str, 'String']):
    dtype = 'U'

    def __init__(
            self, name: str, children: Optional[Sequence['String']] = None,
            categories: Optional[Sequence[str]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None
    ):
        super().__init__(name=name, children=children, categories=categories, nan_freq=nan_freq, num_rows=num_rows)

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> 'String':

        name = cast(str, d["name"])
        extracted = cast(bool, d["extracted"])
        num_rows = cast(Optional[int], d["num_rows"])
        nan_freq = cast(Optional[float], d["nan_freq"])
        categories = cast(Optional[Sequence[str]], d["categories"])
        children: Optional[Sequence[String]] = String.children_from_dict(d)

        meta = cls(name=name, children=children, num_rows=num_rows, nan_freq=nan_freq, categories=categories)
        meta._extracted = extracted

        return meta


class FormattedString(String):
    dtype = 'U'

    def __init__(
            self, name: str, children: Optional[Sequence[String]] = None,
            categories: Optional[Sequence[str]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None, pattern: str = None
    ):
        super().__init__(name=name, children=children, categories=categories, nan_freq=nan_freq, num_rows=num_rows)
        self.pattern = pattern

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "pattern": self.pattern
        })

        return d

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> 'FormattedString':

        name = cast(str, d["name"])
        extracted = cast(bool, d["extracted"])
        num_rows = cast(Optional[int], d["num_rows"])
        nan_freq = cast(Optional[float], d["nan_freq"])
        pattern = cast(Optional[str], d["pattern"])
        categories = cast(Optional[Sequence[str]], d["categories"])
        children: Optional[Sequence[String]] = String.children_from_dict(d)

        meta = cls(
            name=name, children=children, num_rows=num_rows, nan_freq=nan_freq, categories=categories, pattern=pattern
        )
        meta._extracted = extracted

        return meta
