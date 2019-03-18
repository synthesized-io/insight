from .basic import BasicEncoding
from .encoding import Encoding
from .gumbel import GumbelVariationalEncoding
from .rnn_variational import RnnVariationalEncoding
from .variational import VariationalEncoding

encoding_modules = dict(
    basic=BasicEncoding,
    gumbel=GumbelVariationalEncoding,
    rnn_variational=RnnVariationalEncoding,
    variational=VariationalEncoding
)
