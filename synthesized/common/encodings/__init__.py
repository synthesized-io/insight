from .basic import BasicEncoding
from .encoding import Encoding
from .gumbel import GumbelVariationalEncoding
from .variational import VariationalEncoding


encoding_modules = dict(
    basic=BasicEncoding,
    variational=VariationalEncoding,
    gumbel=GumbelVariationalEncoding
)

__all__ = ['Encoding', 'encoding_modules']
