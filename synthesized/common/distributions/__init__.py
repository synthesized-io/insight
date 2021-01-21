from .distribution import Distribution
from ..module import register

register(name='distribution', module=Distribution)


__all__ = ['Distribution']
