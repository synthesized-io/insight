from abc import abstractmethod
from typing import Any, Protocol, TypeVar

DType = TypeVar('DType', covariant=True)

NType = TypeVar("NType", bound='NominalType')
OType = TypeVar("OType", bound='OrdinalType')
AType = TypeVar("AType", bound='AffineType')
SType = TypeVar("SType", bound='ScaleType')
RType = TypeVar("RType", bound='RingType')


class NominalType(Protocol):
    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        pass


class OrdinalType(NominalType):
    @abstractmethod
    def __lt__(self: OType, other: OType) -> bool:
        pass

    def __gt__(self: OType, other: OType) -> bool:
        return (not self < other) and self != other

    def __le__(self: OType, other: OType) -> bool:
        return self < other or self == other

    def __ge__(self: OType, other: OType) -> bool:
        return not self < other


class AffineType(OrdinalType):
    @abstractmethod
    def __sub__(self: AType, other: AType) -> 'ScaleType':
        pass


class ScaleType(AffineType):
    @abstractmethod
    def __add__(self: SType, other: SType) -> 'ScaleType':
        pass


class RingType(ScaleType):
    @abstractmethod
    def __mul__(self: RType, other: RType) -> 'RingType':
        pass
