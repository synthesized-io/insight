from .encoding import Encoding
from .variational import VariationalEncoding
from .rnn_variational import RnnVariationalEncoding
from ..module import register


register(name='variational', module=VariationalEncoding)
register(name='rnn_variational', module=RnnVariationalEncoding)

__all__ = ['Encoding', 'VariationalEncoding', 'RnnVariationalEncoding']
