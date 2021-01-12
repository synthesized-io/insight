from typing import Dict, Type

from .encoding import Encoding
from .recurrent_deep_state_encoding import RecurrentDSSEncoding
from .variational import VariationalEncoding
from .variational_lstm import VariationalLSTMEncoding
from .variational_recurrent_encoding import VariationalRecurrentEncoding
from ..module import register

register(name='recurrent_dss_encoding', module=RecurrentDSSEncoding)
register(name='variational', module=VariationalEncoding)
register(name='variational_lstm', module=VariationalLSTMEncoding)
register(name='vrae', module=VariationalRecurrentEncoding)

Encodings: Dict[str, Type[Encoding]] = {
    'encoding': Encoding,
    'recurrent_dss': RecurrentDSSEncoding,
    'variational': VariationalEncoding,
    'variational_lstm': VariationalLSTMEncoding,
    'vrae': VariationalRecurrentEncoding
}

__all__ = ['Encoding', 'RecurrentDSSEncoding', 'VariationalEncoding', 'VariationalLSTMEncoding',
           'VariationalRecurrentEncoding', 'Encodings']
