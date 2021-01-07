from abc import abstractmethod
from typing import Generic, List, TypeVar

KT = TypeVar('KT')


class Domain(Generic[KT]):
    @abstractmethod
    def __contains__(self, item: KT) -> bool:
        pass

    @abstractmethod
    def tolist(self) -> List[KT]:
        pass
