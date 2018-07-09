from synthesized.core.encodings.encoding import Encoding
from synthesized.core.encodings.basic import BasicEncoding
from synthesized.core.encodings.variational import VariationalEncoding


encoding_modules = dict(
    basic=BasicEncoding,
    variational=VariationalEncoding
)


__all__ = ['encoding_modules', 'Encoding', 'BasicEncoding', 'VariationalEncoding']
