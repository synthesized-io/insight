from synthesized.core.encodings.encoding import Encoding
from synthesized.core.encodings.basic import BasicEncoding
from synthesized.core.encodings.variational import VariationalEncoding
from synthesized.core.encodings.gumbel import GumbelVariationalEncoding


encoding_modules = dict(
    basic=BasicEncoding,
    variational=VariationalEncoding,
    gumbel=GumbelVariationalEncoding
)


__all__ = [
    'encoding_modules', 'Encoding', 'BasicEncoding', 'VariationalEncoding',
    'GumbelVariationalEncoding'
]
