from flask_jwt_extended import get_jwt_identity
from flask_restful import abort

from ..domain.model import Dataset, AccessType
from ..domain.repository import Repository
from typing import Tuple


class DatasetAccessMixin:
    dataset_repo: Repository
    entitlement_repo: Repository

    def get_dataset_authorized(self, dataset_id) -> Dataset:
        dataset = self.dataset_repo.get(dataset_id)
        if not dataset:
            abort(404, message="Couldn't find requested dataset: " + dataset_id)

        entitlements = self.entitlement_repo.find_by_props({'user_id': get_jwt_identity(), 'dataset_id': dataset_id})
        if len(entitlements) > 0:
            entitlement = entitlements[0]
            if entitlement.access_type == AccessType.FULL_ACCESS:
                return dataset

        if dataset.user_id != get_jwt_identity():
            abort(403, message='Dataset with id={} can be accessed only by an owner'.format(dataset_id))

        return dataset

    def get_dataset_access_type(self, dataset_id) -> Tuple[Dataset, AccessType]:
        dataset: Dataset = self.dataset_repo.get(dataset_id)
        if not dataset:
            abort(404, message="Couldn't find requested dataset: " + dataset_id)

        entitlements = self.entitlement_repo.find_by_props({'user_id': get_jwt_identity(), 'dataset_id': dataset_id})
        if len(entitlements) > 0:
            entitlement = entitlements[0]
            return dataset, entitlement.access_type

        if dataset.user_id != get_jwt_identity():
            abort(403, message='Dataset with id={} can be accessed only by an owner'.format(dataset_id))

        return dataset, AccessType.FULL_ACCESS
