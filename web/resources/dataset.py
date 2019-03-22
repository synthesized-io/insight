import os
from io import StringIO, BytesIO

import pandas as pd
import simplejson
from flask import current_app
from flask import jsonify
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restful import Resource, reqparse, abort
from werkzeug.datastructures import FileStorage

from ..domain.analysis import extract_dataset_meta
from ..domain.model import Dataset

SAMPLE_SIZE = 20
MAX_SAMPLE_SIZE = 10000


class DatasetsResource(Resource):
    decorators = [jwt_required]

    def __init__(self, **kwargs):
        self.dataset_repo = kwargs['dataset_repo']

    def get(self):
        datasets = self.dataset_repo.find_by_props({'user_id': get_jwt_identity()})
        return jsonify({
            'datasets': [
                {
                    'dataset_id': d.id,
                    'title': d.title,
                    'description': d.description
                }
                for d in datasets
            ]
        })

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

            meta = extract_dataset_meta(data)
            meta = simplejson.dumps(meta, default=lambda x: x.__dict__, ignore_nan=True).encode('utf-8')

            blob = raw_data.getvalue().encode('utf-8')
            dataset = Dataset(user_id=get_jwt_identity(), title=title, blob=blob, meta=meta)
            self.dataset_repo.save(dataset)
            current_app.logger.info('created a dataset {}'.format(dataset))

            return {'dataset_id': dataset.id}, 201, {'Location': '/datasets/{}'.format(dataset.id)}


class DatasetResource(Resource):
    decorators = [jwt_required]

    def __init__(self, **kwargs):
        self.dataset_repo = kwargs['dataset_repo']

    def get(self, dataset_id):
        parser = reqparse.RequestParser()
        parser.add_argument('sample_size', type=int, default=SAMPLE_SIZE)
        args = parser.parse_args()

        sample_size = args['sample_size']
        if sample_size > MAX_SAMPLE_SIZE:
            abort(400, message='Sample size is too big: ' + str(sample_size))

        dataset = self.dataset_repo.get(dataset_id)
        current_app.logger.info('dataset by id={} is {}'.format(dataset_id, dataset))

        if not dataset:
            abort(404, messsage="Couldn't find requested dataset: " + dataset_id)
        if dataset.user_id != get_jwt_identity():
            abort(403, message='Dataset with id={} can be accessed only by an owner'.format(dataset_id))

        data = pd.read_csv(BytesIO(dataset.blob), encoding='utf-8')

        return jsonify({
            'dataset_id': dataset.id,
            'title': dataset.title,
            'description': dataset.description,
            'meta': simplejson.load(BytesIO(dataset.meta), encoding='utf-8'),
            'sample': data[:sample_size].to_dict(orient='list')
        })

    def delete(self, dataset_id):
        dataset = self.dataset_repo.get(dataset_id)
        current_app.logger.info('dataset by id={} is {}'.format(dataset_id, dataset))
        if dataset:
            if dataset.user_id != get_jwt_identity():
                abort(403, message='Dataset with id={} can be deleted only by an owner'.format(dataset_id))
            self.dataset_repo.delete(dataset)
        return '', 204


class DatasetUpdateInfoResource(Resource):
    decorators = [jwt_required]

    def __init__(self, **kwargs):
        self.dataset_repo = kwargs['dataset_repo']

    def post(self, dataset_id):
        parser = reqparse.RequestParser()
        parser.add_argument('title', type=str, required=False)
        parser.add_argument('description', type=str, required=False)
        args = parser.parse_args()

        dataset = self.dataset_repo.get(dataset_id)
        current_app.logger.info('dataset by id={} is {}'.format(dataset_id, dataset))
        if not dataset:
            abort(404, messsage="Couldn't find requested dataset: " + dataset_id)
        if dataset.user_id != get_jwt_identity():
            abort(403, message='Dataset with id={} can be accessed only by an owner'.format(dataset_id))

        current_app.logger.info('updating the dataset with args {}'.format(args))

        dataset.title = args['title']
        dataset.description = args['description']

        self.dataset_repo.save(dataset)

        return '', 204
