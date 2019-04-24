from abc import ABC, abstractmethod
from typing import Iterable


class EntitlementCompletion(ABC):
    @abstractmethod
    def complete_email(self, creator_id: int, dataset_id: int, q: str) -> Iterable[str]:
        pass
