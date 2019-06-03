from .dense import DenseTransformation
from .mlp import MlpTransformation
from .modulation import ModulationTransformation
from .residual import ResidualTransformation
from .resnet import ResnetTransformation
from .transformation import Transformation

transformation_modules = dict(
    dense=DenseTransformation,
    mlp=MlpTransformation,
    modulation=ModulationTransformation,
    residual=ResidualTransformation,
    resnet=ResnetTransformation
)

__all__ = ['Transformation', 'transformation_modules']
