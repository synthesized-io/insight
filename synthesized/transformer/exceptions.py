"""Transformer Exceptions"""


class TransformerNotFitError(Exception):
    pass


class NonInvertibleTransformError(Exception):
    pass


class UnsupportedMetaError(Exception):
    pass


class FittingError(Exception):
    pass
