from typing import List

from .validity_value import ValidityValue
from ..validity_rules import ValidityRule


class Numerical(ValidityValue):

    def __init__(self, name: str, validity_rules: List[ValidityRule] = None):
        super().__init__(name, validity_rules=validity_rules)
