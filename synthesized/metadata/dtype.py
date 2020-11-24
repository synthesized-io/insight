from typing import TypeVar

import numpy as np

DType = TypeVar('DType', covariant=True)
NType = TypeVar("NType", str, bytes, np.character, np.datetime64, np.integer, np.timedelta64, bool, np.bool8,
                np.float64, covariant=True)
CType = TypeVar("CType", str, bytes, np.character, np.datetime64, np.integer, np.timedelta64, bool, np.bool8,
                np.float64, covariant=True)
OType = TypeVar("OType", np.datetime64, np.integer, np.timedelta64, bool, np.bool8, np.floating, covariant=True)
AType = TypeVar("AType", np.datetime64, np.integer, np.timedelta64, bool, np.bool8, np.floating, covariant=True)
SType = TypeVar("SType", np.integer, np.timedelta64, bool, np.bool8, np.floating, covariant=True)
RType = TypeVar("RType", bool, np.bool8, np.floating, covariant=True)
