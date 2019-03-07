from .middleware import configure_logger
configure_logger()

import os
from collections import namedtuple
from io import StringIO, BytesIO

import pandas as pd
import simplejson
from werkzeug.datastructures import FileStorage
from flask import Flask, jsonify
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_jwt import JWT, jwt_required, current_identity
from flask_restful import Resource, Api, reqparse, abort
from flask_sqlalchemy import SQLAlchemy

from .analisys import extract_dataset_meta, recompute_dataset_meta
from .config import ProductionConfig, DevelopmentConfig
from .middleware import JSONCompliantEncoder

from .repository import SQLAlchemyRepository
from .synthesizer_manager import SynthesizerManager, ModelStatus


SAMPLE_SIZE = 20
MAX_SAMPLE_SIZE = 10000


app = Flask(__name__)
if app.config['ENV'] == 'production':
    app.config.from_object(ProductionConfig)
else:
    app.config.from_object(DevelopmentConfig)
app.json_encoder = JSONCompliantEncoder

CORS(app, supports_credentials=True, send_wildcard=True)  # TODO: delete in final version

db = SQLAlchemy(app)
# models use `db` object, therefore should be imported after
from .model import Dataset, Synthesis, User
db.create_all()

bcrypt = Bcrypt(app)

dataset_repo = SQLAlchemyRepository(db, Dataset)
synthesis_repo = SQLAlchemyRepository(db, Synthesis)
user_repo = SQLAlchemyRepository(db, User)


def authenticate(username, password):
    users = user_repo.find_by_props({'username': username})
    if len(users) > 0:
        password_hash = bytes.fromhex(users[0].password)
        if bcrypt.check_password_hash(password_hash, password):
            return users[0]


def identity(payload):
    user_id = payload['identity']
    return user_repo.get(user_id)


jwt = JWT(app, authenticate, identity)


class UsersResource(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('username', type=str, required=True)
        parser.add_argument('password', type=str, required=True)
        args = parser.parse_args()

        username = args['username']
        password = args['password']

        app.logger.info('registering user {}'.format(username))

        users = user_repo.find_by_props({'username': username})
        if len(users) > 0:
            app.logger.info('found existing user {}'.format(users[0]))
            abort(409, message='User with username={} already exists'.format(username))

        user = User(username=username, password=bcrypt.generate_password_hash(password).hex())
        user_repo.save(user)
        app.logger.info('created a user {}'.format(user))

        return {'user_id': user.id}, 201


class DatasetsResource(Resource):
    decorators = [jwt_required()]

    def get(self):
        datasets = dataset_repo.find_by_props({'user_id': current_identity.id})
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

        app.logger.info('processing file {}'.format(file.filename))

        with file.stream as stream:
            data = pd.read_csv(stream)

            title = os.path.splitext(file.filename)[0]

            raw_data = StringIO()
            data.to_csv(raw_data, index=False, encoding='utf-8')

            meta = extract_dataset_meta(data)
            meta = simplejson.dumps(meta, default=lambda x: x.__dict__, ignore_nan=True).encode('utf-8')

            blob = raw_data.getvalue().encode('utf-8')
            dataset = Dataset(user_id=current_identity.id, title=title, blob=blob, meta=meta)
            dataset_repo.save(dataset)
            app.logger.info('created a dataset {}'.format(dataset))

            return {'dataset_id': dataset.id}, 201


class DatasetResource(Resource):
    decorators = [jwt_required()]

    def get(self, dataset_id):
        parser = reqparse.RequestParser()
        parser.add_argument('sample_size', type=int, default=SAMPLE_SIZE)
        args = parser.parse_args()

        sample_size = args['sample_size']
        if sample_size > MAX_SAMPLE_SIZE:
            abort(400, message='Sample size is too big: ' + str(sample_size))

        dataset = dataset_repo.get(dataset_id)
        app.logger.info('dataset by id={} is {}'.format(dataset_id, dataset))

        if not dataset:
            abort(404, messsage="Couldn't find requested dataset: " + dataset_id)
        if dataset.user_id != current_identity.id:
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
        dataset = dataset_repo.get(dataset_id)
        app.logger.info('dataset by id={} is {}'.format(dataset_id, dataset))
        if dataset:
            if dataset.user_id != current_identity.id:
                abort(403, message='Dataset with id={} can be deleted only by an owner'.format(dataset_id))
            dataset_repo.delete(dataset)
        return '', 204


class DatasetUpdateInfoResource(Resource):
    decorators = [jwt_required()]

    def post(self, dataset_id):
        parser = reqparse.RequestParser()
        parser.add_argument('title', type=str, required=False)
        parser.add_argument('description', type=str, required=False)
        args = parser.parse_args()

        dataset = dataset_repo.get(dataset_id)
        app.logger.info('dataset by id={} is {}'.format(dataset_id, dataset))
        if not dataset:
            abort(404, messsage="Couldn't find requested dataset: " + dataset_id)

        app.logger.info('updating the dataset with args {}'.format(args))

        dataset.title = args['title']
        dataset.description = args['description']

        dataset_repo.save(dataset)

        return '', 204


synthesizer_manager = SynthesizerManager(dataset_repo=dataset_repo)


class ModelResource(Resource):
    decorators = [jwt_required()]

    def get(self, dataset_id):
        model = synthesizer_manager.get_model(dataset_id)
        if not model:
            abort(404, message='Model does not exist for dataset={}'.format(dataset_id))

        return {'model': str(model)}, 200

    def post(self, dataset_id):
        dataset = dataset_repo.get(dataset_id)
        app.logger.info('dataset by id={} is {}'.format(dataset_id, dataset))
        if not dataset:
            abort(404, message="Couldn't find requested dataset: " + dataset_id)
        if dataset.user_id != current_identity.id:
            abort(403, message='Dataset with id={} can be accessed only by an owner'.format(dataset_id))

        synthesizer_manager.train_async(dataset_id)

        return {'status': 'training'}, 202, {'Location': '/datasets/{}/model-training'.format(dataset_id)}


class ModelTrainingResource(Resource):
    decorators = [jwt_required()]

    def get(self, dataset_id):
        status = synthesizer_manager.get_status(dataset_id)
        if status == ModelStatus.NO_MODEL:
            abort(404, message="No training for dataset_id " + dataset_id)
        if status == ModelStatus.TRAINING:
            return {'status': 'training'}, 200

        return {'status': 'ready'}, 303, {'Location': '/datasets/{}/model'.format(dataset_id)}


class SynthesesResource(Resource):
    decorators = [jwt_required()]

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('dataset_id', type=str, required=True)
        parser.add_argument('rows', type=int, required=True)
        args = parser.parse_args()

        dataset_id = args['dataset_id']
        rows = args['rows']

        app.logger.info('synthesis for dataset_id={}'.format(dataset_id))

        dataset = dataset_repo.get(dataset_id)
        app.logger.info('dataset by id={} is {}'.format(dataset_id, dataset))
        if not dataset:
            abort(404, message="Couldn't find requested dataset: " + dataset_id)

        model = synthesizer_manager.get_model(dataset_id)
        app.logger.info('found a model {}'.format(model))
        if not model:
            abort(404, message='Model does not exist for user={} and dataset={}'.format(current_identity.id, dataset_id))

        app.logger.info('starting synthesis')
        synthesized = model.synthesize(rows)

        output = StringIO()
        synthesized.to_csv(output, index=False, encoding='utf-8')

        blob = output.getvalue().encode('utf-8')

        # Parse JSON into an object with attributes corresponding to dict keys.
        original_meta = simplejson.load(BytesIO(dataset.meta), encoding='utf-8', object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

        synthetic_meta = recompute_dataset_meta(synthesized, original_meta)
        synthetic_meta = simplejson.dumps(synthetic_meta, default=lambda x: x.__dict__, ignore_nan=True).encode('utf-8')

        synthesis = Synthesis(dataset_id=dataset_id, blob=blob, meta=synthetic_meta, size=rows)
        synthesis_repo.save(synthesis)

        app.logger.info('created a synthesis {}'.format(synthesis))

        return {'synthesis_id': synthesis.id}, 201


class SynthesisResource(Resource):
    decorators = [jwt_required()]

    def get(self, synthesis_id):
        parser = reqparse.RequestParser()
        parser.add_argument('sample_size', type=int, default=SAMPLE_SIZE)
        args = parser.parse_args()

        sample_size = args['sample_size']
        if sample_size > MAX_SAMPLE_SIZE:
            abort(400, message='Sample size is too big: ' + str(sample_size))

        synthesis = synthesis_repo.get(synthesis_id)
        app.logger.info('synthesis by id={} is {}'.format(synthesis_id, synthesis))

        if not synthesis:
            abort(404, messsage="Couldn't find requested synthesis: " + synthesis_id)

        data = pd.read_csv(BytesIO(synthesis.blob))

        return jsonify({
            'synthesis_id': synthesis_id,
            'meta': simplejson.load(BytesIO(synthesis.meta), encoding='utf-8'),
            'sample': data[:sample_size].to_dict(orient='list')
        })


class StatusResource(Resource):
    def get(self):
        return {'success': True}


api = Api(app)
api.add_resource(StatusResource, '/')
api.add_resource(UsersResource, '/users')
api.add_resource(DatasetsResource, '/datasets')
api.add_resource(DatasetResource, '/datasets/<dataset_id>')
api.add_resource(DatasetUpdateInfoResource, '/datasets/<dataset_id>/updateinfo')
api.add_resource(ModelResource, '/datasets/<dataset_id>/model')
api.add_resource(ModelTrainingResource, '/datasets/<dataset_id>/model-training')
api.add_resource(SynthesesResource, '/syntheses')
api.add_resource(SynthesisResource, '/syntheses/<synthesis_id>')
