from .encoding import Encoding
from .variational import VariationalEncoding
from ..module import register


register(name='variational', module=VariationalEncoding)

__all__ = ['Encoding']
