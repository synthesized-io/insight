from .encoding import Encoding
from .variational import VariationalEncoding
from .variational_lstm import VariationalLSTMEncoding
from .variational_recurrent_encoding import VariationalRecurrentEncoding
from ..module import register


register(name='variational', module=VariationalEncoding)
register(name='variational_lstm', module=VariationalLSTMEncoding)
register(name='vrae', module=VariationalRecurrentEncoding)

__all__ = ['Encoding', 'VariationalEncoding', 'VariationalLSTMEncoding', 'VariationalRecurrentEncoding']
