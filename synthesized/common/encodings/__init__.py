from .encoding import Encoding
from .rnn_variational import RnnVariationalEncoding
from .variational import VariationalEncoding
from .feedforward_state_space import FeedForwardDSSMEncoding
from ..module import register


register(name='rnn_variational', module=RnnVariationalEncoding)
register(name='variational', module=VariationalEncoding)

__all__ = ['Encoding']
