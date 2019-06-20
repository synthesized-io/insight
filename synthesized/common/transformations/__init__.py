from .dense import DenseTransformation
from .linear import LinearTransformation
from .mlp import MlpTransformation
from .modulation import ModulationTransformation
from .residual import ResidualTransformation
from .resnet import ResnetTransformation
from .transformation import Transformation

transformation_modules = dict(
    dense=DenseTransformation,
    linear=LinearTransformation,
    mlp=MlpTransformation,
    modulation=ModulationTransformation,
    residual=ResidualTransformation,
    resnet=ResnetTransformation
)

__all__ = ['Transformation', 'transformation_modules']
