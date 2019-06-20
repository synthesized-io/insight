from .dense import DenseTransformation
from .linear import LinearTransformation
from .mlp import MlpTransformation
from .modulation import ModulationTransformation
from .residual import ResidualTransformation
from .resnet import ResnetTransformation
from .transformation import Transformation
from ..module import register


register(name='dense', module=DenseTransformation)
register(name='linear', module=LinearTransformation)
register(name='mlp', module=MlpTransformation)
register(name='modulation', module=ModulationTransformation)
register(name='residual', module=ResidualTransformation)
register(name='resnet', module=ResnetTransformation)


__all__ = ['Transformation']
