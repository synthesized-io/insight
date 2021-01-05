from ..module import register
from .distribution import Distribution

register(name='distribution', module=Distribution)


__all__ = ['Distribution']
