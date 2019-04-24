from typing import Iterable

from flask_sqlalchemy import SQLAlchemy

from ..application.entitelement_completion import EntitlementCompletion
from ..domain.model import User, Entitlement


class SQLAlchemyEntitlementCompletion(EntitlementCompletion):
    def __init__(self, db: SQLAlchemy):
        self.db = db

    def complete_email(self, creator_id: int, dataset_id: int, q: str) -> Iterable[str]:
        """
        Rules:

        * include all user emails starting with `q`
        * exclude already entitled users
        * exclude creator
        * order alphabetically
        """
        pattern = '{}%'.format(q)
        entitled_user_ids = self.db.session.query(Entitlement.user_id) \
            .filter(Entitlement.creator_id == creator_id) \
            .filter(Entitlement.dataset_id == dataset_id)
        return [email for email, in
                self.db.session.query(User.email)
                    .filter(User.email.ilike(pattern))
                    .filter(User.id.notin_(entitled_user_ids))
                    .filter(User.id != creator_id)
                    .order_by(User.email).all()]
