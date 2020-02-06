from .encoding import Encoding
from .variational import VariationalEncoding
from .rnn_variational import RnnVariationalEncoding
from .feedforward_state_space import FeedForwardDSSMEncoding
from .recurrent_state_space import RecurrentDSSMEncoding
from ..module import register


register(name='variational', module=VariationalEncoding)

__all__ = ['Encoding', 'VariationalEncoding']
