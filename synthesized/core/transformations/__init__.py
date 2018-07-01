from synthesized.core.transformations.transformation import Transformation
from synthesized.core.transformations.mlp import MlpTransformation


transformation_modules = dict(
    mlp=MlpTransformation
)


__all__ = ['transformation_modules', 'Transformation', 'MlpTransformation']
