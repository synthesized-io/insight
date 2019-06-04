from .conditional import ConditionalFunctional
from .correlation import CorrelationFunctional
from .correlation_matrix import CorrelationMatrixFunctional
from .functional import Functional
from .mean import MeanFunctional
from .standard_deviation import StandardDeviationFunctional

functional_modules = dict(
    conditional=ConditionalFunctional,
    correlation=CorrelationFunctional,
    correlation_matrix=CorrelationMatrixFunctional,
    mean=MeanFunctional,
    stddev=StandardDeviationFunctional
)

__all__ = ['Functional', 'functional_modules']
