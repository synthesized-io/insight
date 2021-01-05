from ..module import register
from .conditional import ConditionalFunctional
from .correlation import CorrelationFunctional
from .correlation_matrix import CorrelationMatrixFunctional
from .functional import Functional
from .mean import MeanFunctional
from .standard_deviation import StandardDeviationFunctional

register(name='conditional', module=ConditionalFunctional)
register(name='correlation', module=CorrelationFunctional)
register(name='correlation_matrix', module=CorrelationMatrixFunctional)
register(name='mean', module=MeanFunctional)
register(name='stddev', module=StandardDeviationFunctional)


__all__ = ['Functional']
