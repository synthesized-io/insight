from ..module import register
from .dense import DenseTransformation
from .gaussian_transformation import GaussianTransformation
from .linear import LinearTransformation
from .lstm import LstmTransformation
from .mlp import MlpTransformation
from .modulation import ModulationTransformation
from .residual import ResidualTransformation
from .resnet import ResnetTransformation
from .transformation import Transformation

register(name='dense', module=DenseTransformation)
register(name='gaussian_transformation', module=GaussianTransformation)
register(name='linear', module=LinearTransformation)
register(name='lstm', module=LstmTransformation)
register(name='mlp', module=MlpTransformation)
register(name='modulation', module=ModulationTransformation)
register(name='residual', module=ResidualTransformation)
register(name='resnet', module=ResnetTransformation)


__all__ = ['Transformation', 'DenseTransformation', 'GaussianTransformation', 'LinearTransformation',
           'MlpTransformation', 'ModulationTransformation', 'ResidualTransformation', 'ResnetTransformation']
