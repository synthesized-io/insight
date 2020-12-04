"""
Custom exception classes.

This module contains custom exceptions used by the Synthesized SDK.
"""


class MetaNotExtractedError(Exception):
    pass


class ExtractionError(Exception):
    pass


class UnsupportedDtypeError(Exception):
    pass


class UnknownDateFormatError(Exception):
    pass


class TransformerNotFitError(Exception):
    pass


class NonInvertibleTransformError(Exception):
    pass


class UnsupportedMetaError(Exception):
    pass
