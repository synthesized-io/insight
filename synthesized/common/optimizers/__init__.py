from ..module import register
from .optimizer import Optimizer

register(name='optimizer', module=Optimizer)


__all__ = ['Optimizer']
