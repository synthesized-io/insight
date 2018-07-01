from synthesized.core.values.value import Value
from synthesized.core.values.categorical import CategoricalValue
from synthesized.core.values.continuous import ContinuousValue


value_modules = dict(
    categorical=CategoricalValue,
    continuous=ContinuousValue
)


__all__ = ['value_modules', 'Value', 'CategoricalValue', 'ContinuousValue']
