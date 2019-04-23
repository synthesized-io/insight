from flask_jwt_extended import get_jwt_identity
from flask_restful import abort

from ..domain.model import Dataset
from ..domain.repository import Repository


class DatasetAccessMixin:
    dataset_repo: Repository

    def get_dataset_authorized(self, dataset_id) -> Dataset:
        dataset = self.dataset_repo.get(dataset_id)
        if not dataset:
            abort(404, message="Couldn't find requested dataset: " + dataset_id)
        if dataset.user_id != get_jwt_identity():
            abort(403, message='Dataset with id={} can be accessed only by an owner'.format(dataset_id))
        return dataset