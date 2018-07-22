from synthesized.core.values.value import Value
from synthesized.core.values.categorical import CategoricalValue
from synthesized.core.values.continuous import ContinuousValue
from synthesized.core.values.identifier import IdentifierValue


value_modules = dict(
    categorical=CategoricalValue,
    continuous=ContinuousValue,
    identifier=IdentifierValue
)


__all__ = ['value_modules', 'Value', 'CategoricalValue', 'ContinuousValue', 'IdentifierValue']
