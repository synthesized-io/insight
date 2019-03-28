from abc import ABC, abstractmethod


class ReportItemOrdering(ABC):
    @abstractmethod
    def move_item(self, report_item_id, new_order):
        pass
