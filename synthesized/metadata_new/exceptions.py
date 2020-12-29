"""Custom exception classes relating to DataMeta.

This module contains custom exceptions used by the Synthesized SDK.
"""


class MetaNotExtractedError(Exception):
    pass


class ModelNotFittedError(Exception):
    pass


class ExtractionError(Exception):
    pass


class UnsupportedDtypeError(Exception):
    pass


class UnknownDateFormatError(Exception):
    pass
