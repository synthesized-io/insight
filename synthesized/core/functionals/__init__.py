from .correlation import CorrelationFunctional
from .correlation_matrix import CorrelationMatrixFunctional
from .functional import Functional
from .mean import MeanFunctional
from .variance import VarianceFunctional

functional_modules = dict(
    correlation=CorrelationFunctional,
    correlation_matrix=CorrelationMatrixFunctional,
    mean=MeanFunctional,
    variance=VarianceFunctional
)
