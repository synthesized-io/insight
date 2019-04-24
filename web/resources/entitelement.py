from typing import Iterable

from flask import jsonify
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restful import Resource, reqparse, abort

from .common import DatasetAccessMixin
from ..domain.model import Entitlement
from ..domain.repository import Repository


class EntitlementsResource(Resource, DatasetAccessMixin):
    decorators = [jwt_required]

    def __init__(self, **kwargs):
        self.dataset_repo: Repository = kwargs['dataset_repo']
        self.user_repo: Repository = kwargs['user_repo']
        self.entitlement_repo: Repository = kwargs['entitlement_repo']

    def get(self, dataset_id):
        self.get_dataset_authorized(dataset_id)
        entitlements: Iterable[Entitlement] = self.entitlement_repo.find_by_props({'creator_id': get_jwt_identity(), 'dataset_id': dataset_id})
        return jsonify({
            'entitlements': [
                {
                    'entitlement_id': e.id,
                    'email': e.user.email,
                    'access_type': e.access_type.name
                }
                for e in entitlements
            ]
        })

    def post(self, dataset_id):
        self.get_dataset_authorized(dataset_id)

        parser = reqparse.RequestParser()
        parser.add_argument('email', type=str, required=False)
        parser.add_argument('access_type', type=str, choices=('FULL_ACCESS', 'RESULTS_ONLY'), required=False)
        args = parser.parse_args()

        email = args['email']
        access_type = args['access_type']

        users = self.user_repo.find_by_props({'email': email})
        if len(users) == 0:
            abort(409, message='Can not find user by email ' + email)
        user = users[0]

        if get_jwt_identity() == user.id:
            abort(409, message='User can not entitle themself')

        existing = self.entitlement_repo.find_by_props({'creator_id': get_jwt_identity(), 'dataset_id': dataset_id, 'user_id': user.id})
        if len(existing) > 0:
            abort(409, message='Entitlement has already been created for ' + email)

        entitlement = Entitlement(creator_id=get_jwt_identity(), dataset_id=dataset_id, user_id=user.id, access_type=access_type)
        self.entitlement_repo.save(entitlement)

        return {'entitlement_id': entitlement.id}, 201


class EntitlementResource(Resource, DatasetAccessMixin):
    decorators = [jwt_required]

    def __init__(self, **kwargs):
        self.dataset_repo: Repository = kwargs['dataset_repo']
        self.entitlement_repo: Repository = kwargs['entitlement_repo']

    def put(self, dataset_id, entitlement_id):
        self.get_dataset_authorized(dataset_id)

        entitlement = self.entitlement_repo.get(entitlement_id)
        if not entitlement:
            abort(404, message='Could not find entitlement ' + str(entitlement_id))

        parser = reqparse.RequestParser()
        parser.add_argument('access_type', type=str, choices=('FULL_ACCESS', 'RESULTS_ONLY'), required=False)
        args = parser.parse_args()

        access_type = args['access_type']
        entitlement.access_type = access_type
        self.entitlement_repo.save(entitlement)

        return '', 204

    def delete(self, dataset_id, entitlement_id):
        self.get_dataset_authorized(dataset_id)

        entitlement = self.entitlement_repo.get(entitlement_id)
        if not entitlement:
            abort(404, message='Could not find entitlement ' + str(entitlement_id))

        self.entitlement_repo.delete(entitlement)

        return '', 204
