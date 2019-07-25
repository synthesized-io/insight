from .optimizer import Optimizer
from ..module import register


register(name='optimizer', module=Optimizer)


__all__ = ['Optimizer']