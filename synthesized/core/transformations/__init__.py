from synthesized.core.transformations.transformation import Transformation
from synthesized.core.transformations.dense import DenseTransformation
from synthesized.core.transformations.mlp import MlpTransformation
from synthesized.core.transformations.modulation import ModulationTransformation


transformation_modules = dict(
    dense=DenseTransformation,
    mlp=MlpTransformation,
    modulation=ModulationTransformation
)


__all__ = [
    'transformation_modules', 'Transformation', 'DenseTransformation', 'MlpTransformation',
    'ModulationTransformation'
]
