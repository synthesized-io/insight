from .dense import DenseTransformation
from .mlp import MlpTransformation
from .modulation import ModulationTransformation
from .transformation import Transformation

transformation_modules = dict(
    dense=DenseTransformation,
    mlp=MlpTransformation,
    modulation=ModulationTransformation
)
