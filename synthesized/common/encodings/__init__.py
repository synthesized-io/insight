from .encoding import Encoding
from .variational import VariationalEncoding
from .feedforward_state_space import FeedForwardDSSMEncoding
from ..module import register


register(name='variational', module=VariationalEncoding)

__all__ = ['Encoding', 'VariationalEncoding']
