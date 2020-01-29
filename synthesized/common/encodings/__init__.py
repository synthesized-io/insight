from .encoding import Encoding
from .rnn_variational_new import RnnVariationalEncoding
from .variational import VariationalEncoding
from ..module import register


register(name='rnn_variational', module=RnnVariationalEncoding)
register(name='variational', module=VariationalEncoding)

__all__ = ['Encoding']
