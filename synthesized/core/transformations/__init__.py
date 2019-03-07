from .dense import DenseTransformation
from .lstm import LstmTransformation
from .mlp import MlpTransformation
from .modulation import ModulationTransformation
from .residual import ResidualTransformation
from .resnet import ResnetTransformation
from .transformation import Transformation

transformation_modules = dict(
    dense=DenseTransformation,
    lstm=LstmTransformation,
    mlp=MlpTransformation,
    modulation=ModulationTransformation,
    residual=ResidualTransformation,
    resnet=ResnetTransformation
)
