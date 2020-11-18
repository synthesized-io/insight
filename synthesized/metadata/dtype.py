from abc import abstractmethod
from typing import Any, TypeVar
from typing_extensions import Protocol

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


class OrdinalType(NominalType, Protocol):
    @abstractmethod
    def __lt__(self: OType, other: OType) -> bool:
        pass

    @abstractmethod
    def __gt__(self: OType, other: OType) -> bool:
        pass

    @abstractmethod
    def __le__(self: OType, other: OType) -> bool:
        pass

    @abstractmethod
    def __ge__(self: OType, other: OType) -> bool:
        pass


class AffineType(OrdinalType, Protocol):
    @abstractmethod
    def __sub__(self: AType, other: AType) -> 'ScaleType':
        pass


class ScaleType(AffineType, Protocol):
    @abstractmethod
    def __add__(self: SType, other: SType) -> 'ScaleType':
        pass


class RingType(ScaleType, Protocol):
    @abstractmethod
    def __mul__(self: RType, other: RType) -> 'RingType':
        pass
