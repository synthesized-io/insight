from typing import Dict, Optional, Sequence

from ..base import Nominal, ValueMeta


class String(Nominal[str]):
    dtype = 'U'

    def __init__(
            self, name: str, children: Optional[Sequence[ValueMeta]] = None,
            categories: Optional[Sequence[str]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None
    ):
        super().__init__(name=name, children=children, categories=categories, nan_freq=nan_freq, num_rows=num_rows)


class FormattedString(String):
    dtype = 'U'

    def __init__(
            self, name: str, categories: Optional[Sequence[str]] = None, nan_freq: Optional[float] = None,
            num_rows: Optional[int] = None, pattern: str = None
    ):
        super().__init__(name=name, categories=categories, nan_freq=nan_freq, num_rows=num_rows)
        self.pattern = pattern

    def to_dict(self) -> Dict[str, object]:
        d = super().to_dict()
        d.update({
            "pattern": self.pattern
        })

        return d
