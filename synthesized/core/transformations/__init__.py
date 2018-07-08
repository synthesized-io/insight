from synthesized.core.transformations.transformation import Transformation
from synthesized.core.transformations.dense import DenseTransformation
from synthesized.core.transformations.mlp import MlpTransformation


transformation_modules = dict(
    dense=DenseTransformation,
    mlp=MlpTransformation
)


__all__ = ['transformation_modules', 'Transformation', 'DenseTransformation', 'MlpTransformation']
