import logging

from flask_sqlalchemy import SQLAlchemy

from ..application.report_item_ordering import ReportItemOrdering
from ..domain.model import ReportItem

logger = logging.getLogger(__name__)


class SQLAlchemyReportItemOrdering(ReportItemOrdering):
    def __init__(self, db: SQLAlchemy):
        self.db = db

    def move_item(self, report_item_id, new_order):
        report_item: ReportItem = self.db.session.query(ReportItem).get(report_item_id)
        if not report_item:
            logger.warning('item not found: {}'.format(report_item_id))
            return

        if report_item.ord > new_order:
            start = new_order
            end = report_item.ord - 1
            logger.info('from pos {} to pos {} increasing the index'.format(start, end))
            self.db.session \
                .query(ReportItem)\
                .filter(ReportItem.report_id == report_item.report_id) \
                .filter(ReportItem.ord >= start) \
                .filter(ReportItem.ord <= end) \
                .update({ReportItem.ord: ReportItem.ord + 1})
        elif report_item.ord < new_order:
            start = report_item.ord + 1
            end = new_order
            logger.info('from pos {} to pos {} decreasing the index'.format(start, end))
            self.db.session \
                .query(ReportItem) \
                .filter(ReportItem.report_id == report_item.report_id) \
                .filter(ReportItem.ord >= start) \
                .filter(ReportItem.ord <= end) \
                .update({ReportItem.ord: ReportItem.ord - 1})

        logger.info("moving item from {} to {}".format(report_item.ord, new_order))
        report_item.ord = new_order

        self.db.session.commit()
