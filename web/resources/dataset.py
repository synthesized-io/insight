import os
import re
from datetime import datetime
from io import StringIO, BytesIO
from operator import itemgetter
from typing import Iterable

import pandas as pd
import simplejson
from flask import current_app, jsonify, send_file
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restful import Resource, reqparse, abort
from werkzeug.datastructures import FileStorage

from .common import DatasetAccessMixin
from ..domain.dataset_meta import compute_dataset_meta
from ..domain.model import Dataset, Entitlement
from ..domain.repository import Repository

SAMPLE_SIZE = 20
MAX_SAMPLE_SIZE = 10000


class DatasetsResource(Resource):
    decorators = [jwt_required]

    def __init__(self, **kwargs):
        self.dataset_repo: Repository = kwargs['dataset_repo']
        self.entitlement_repo: Repository = kwargs['entitlement_repo']

    def get(self):
        datasets: Iterable[Dataset] = self.dataset_repo.find_by_props({'user_id': get_jwt_identity()})
        entitlements: Iterable[Entitlement] = self.entitlement_repo.find_by_props({'user_id': get_jwt_identity()})
        datasets_json = []
        datasets_json.extend([
            {
                'dataset_id': d.id,
                'title': d.title,
                'description': d.description
            }
            for d in datasets
        ])
        datasets_json.extend([
            {
                'dataset_id': e.dataset.id,
                'title': e.dataset.title,
                'description': e.dataset.description,
                'shared_by': e.creator.email,
            }
            for e in entitlements
        ])
        datasets_json = sorted(datasets_json, key=itemgetter('dataset_id'))
        return jsonify({'datasets': datasets_json})

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('file', type=FileStorage, location='files', required=True)
        args = parser.parse_args()
        file = args['file']

        current_app.logger.info('processing file {}'.format(file.filename))

        with file.stream as stream:
            data = pd.read_csv(stream)

            title = os.path.splitext(file.filename)[0]

            raw_data = StringIO()
            data.to_csv(raw_data, index=False, encoding='utf-8')

            meta = compute_dataset_meta(data)
            blob = raw_data.getvalue().encode('utf-8')

            dataset = Dataset(user_id=get_jwt_identity(), title=title, blob=blob)
            dataset.set_meta_from_object(meta)

            self.dataset_repo.save(dataset)
            current_app.logger.info('created a dataset {}'.format(dataset))

            return {'dataset_id': dataset.id}, 201, {'Location': '/datasets/{}'.format(dataset.id)}


class DatasetResource(Resource, DatasetAccessMixin):
    decorators = [jwt_required]

    def __init__(self, **kwargs):
        self.dataset_repo: Repository = kwargs['dataset_repo']
        self.entitlement_repo: Repository = kwargs['entitlement_repo']

    def get(self, dataset_id):
        dataset = self.get_dataset_authorized(dataset_id)

        parser = reqparse.RequestParser()
        parser.add_argument('sample_size', type=int, default=SAMPLE_SIZE)
        args = parser.parse_args()

        sample_size = args['sample_size']
        if sample_size > MAX_SAMPLE_SIZE:
            abort(400, message='Sample size is too big: ' + str(sample_size))

        data = pd.read_csv(BytesIO(dataset.blob), encoding='utf-8')

        return jsonify({
            'dataset_id': dataset.id,
            'title': dataset.title,
            'description': dataset.description,
            'meta': simplejson.load(BytesIO(dataset.meta), encoding='utf-8'),
            'sample': data[:sample_size].to_dict(orient='list'),
            'settings': dataset.get_settings_as_dict(),
        })

    def delete(self, dataset_id):
        dataset = self.dataset_repo.get(dataset_id)
        current_app.logger.info('dataset by id={} is {}'.format(dataset_id, dataset))
        if dataset:
            if dataset.user_id != get_jwt_identity():
                abort(403, message='Dataset with id={} can be deleted only by an owner'.format(dataset_id))
            self.dataset_repo.delete(dataset)
        return '', 204


class DatasetUpdateInfoResource(Resource, DatasetAccessMixin):
    decorators = [jwt_required]

    def __init__(self, **kwargs):
        self.dataset_repo: Repository = kwargs['dataset_repo']
        self.entitlement_repo: Repository = kwargs['entitlement_repo']

    def post(self, dataset_id):
        dataset = self.get_dataset_authorized(dataset_id)

        parser = reqparse.RequestParser()
        parser.add_argument('title', type=str, required=False)
        parser.add_argument('description', type=str, required=False)
        args = parser.parse_args()

        current_app.logger.info('updating the dataset with args {}'.format(args))

        dataset.title = args['title']
        dataset.description = args['description']

        self.dataset_repo.save(dataset)

        return '', 204


class DatasetUpdateSettingsResource(Resource, DatasetAccessMixin):
    decorators = [jwt_required]

    def __init__(self, **kwargs):
        self.dataset_repo: Repository = kwargs['dataset_repo']
        self.entitlement_repo: Repository = kwargs['entitlement_repo']

    def post(self, dataset_id):
        dataset = self.get_dataset_authorized(dataset_id)

        parser = reqparse.RequestParser()
        parser.add_argument('settings', type=dict, required=False)
        args = parser.parse_args()

        settings = args['settings']
        dataset.settings = simplejson.dumps(settings).encode('utf-8')

        self.dataset_repo.save(dataset)

        return '', 204


class DatasetExportResource(Resource, DatasetAccessMixin):
    decorators = [jwt_required]

    def __init__(self, **kwargs):
        self.dataset_repo: Repository = kwargs['dataset_repo']
        self.synthesis_repo: Repository = kwargs['synthesis_repo']
        self.entitlement_repo: Repository = kwargs['entitlement_repo']

    def get(self, dataset_id):
        dataset, _ = self.get_dataset_access_type(dataset_id)

        syntheses = self.synthesis_repo.find_by_props({'dataset_id': dataset_id})
        current_app.logger.info('synthesis by dataset_id={} is {}'.format(dataset_id, syntheses))

        if len(syntheses) == 0:
            abort(404, messsage="Couldn't find requested synthesis")

        synthesis = syntheses[0]  # assumes the only one synthesis

        filename = ''
        if dataset.title:
            filename += re.sub('[^0-9a-zA-Z]+', '_', dataset.title).lower() + '_'
        filename += 'synthesized_'
        filename += datetime.today().strftime('%Y%m%d')
        filename += '.csv'
        return send_file(BytesIO(synthesis.blob), mimetype='text/csv', as_attachment=True, attachment_filename=filename)
