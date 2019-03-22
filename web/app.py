
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
from flask_restful import Resource, Api, reqparse, abort
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, create_refresh_token, get_jwt_identity, \
    jwt_refresh_token_required

from .analysis import extract_dataset_meta, recompute_dataset_meta
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

CORS(app, supports_credentials=True)  # TODO: delete in final version

db = SQLAlchemy(app)
# models use `db` object, therefore should be imported after
from .model import Dataset, Synthesis, User
db.create_all()

bcrypt = Bcrypt(app)


class Authenticator:
    def __init__(self, user_repo):
        self.user_repo = user_repo

    def authenticate(self, username, password):
        users = self.user_repo.find_by_props({'username': username})
        if len(users) > 0:
            password_hash = bytes.fromhex(users[0].password)
            if bcrypt.check_password_hash(password_hash, password):
                return users[0]


jwt = JWTManager(app)


class LoginResource(Resource):
    def __init__(self, **kwargs):
        self.authenticator = kwargs['authenticator']

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('username', type=str, required=True)
        parser.add_argument('password', type=str, required=True)
        args = parser.parse_args()

        username = args['username']
        password = args['password']

        user = self.authenticator.authenticate(username, password)
        if not user:
            return {"message": "Bad username or password"}, 401

        ret = {
            'access_token': create_access_token(identity=user.id),
            'refresh_token': create_refresh_token(identity=user.id)
        }
        return ret, 200


class RefreshResource(Resource):
    decorators = [jwt_refresh_token_required]

    def post(self):
        current_user = get_jwt_identity()
        ret = {
            'access_token': create_access_token(identity=current_user)
        }
        return ret, 200


class UsersResource(Resource):
    def __init__(self, **kwargs):
        self.user_repo = kwargs['user_repo']

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('username', type=str, required=True)
        parser.add_argument('password', type=str, required=True)
        args = parser.parse_args()

        username = args['username']
        password = args['password']

        app.logger.info('registering user {}'.format(username))

        users = self.user_repo.find_by_props({'username': username})
        if len(users) > 0:
            app.logger.info('found existing user {}'.format(users[0]))
            abort(409, message='User with username={} already exists'.format(username))

        user = User(username=username, password=bcrypt.generate_password_hash(password).hex())
        self.user_repo.save(user)
        app.logger.info('created a user {}'.format(user))

        ret = {
            'access_token': create_access_token(identity=user.id),
            'refresh_token': create_refresh_token(identity=user.id)
        }
        return ret, 200


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

        app.logger.info('processing file {}'.format(file.filename))

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
            app.logger.info('created a dataset {}'.format(dataset))

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
        app.logger.info('dataset by id={} is {}'.format(dataset_id, dataset))

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
        app.logger.info('dataset by id={} is {}'.format(dataset_id, dataset))
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
        app.logger.info('dataset by id={} is {}'.format(dataset_id, dataset))
        if not dataset:
            abort(404, messsage="Couldn't find requested dataset: " + dataset_id)
        if dataset.user_id != get_jwt_identity():
            abort(403, message='Dataset with id={} can be accessed only by an owner'.format(dataset_id))

        app.logger.info('updating the dataset with args {}'.format(args))

        dataset.title = args['title']
        dataset.description = args['description']

        self.dataset_repo.save(dataset)

        return '', 204


class ModelResource(Resource):
    decorators = [jwt_required]

    def __init__(self, **kwargs):
        self.dataset_repo = kwargs['dataset_repo']
        self.synthesizer_manager = kwargs['synthesizer_manager']

    def get(self, dataset_id):
        dataset = self.dataset_repo.get(dataset_id)
        app.logger.info('dataset by id={} is {}'.format(dataset_id, dataset))
        if not dataset:
            abort(404, message="Couldn't find requested dataset: " + dataset_id)
        if dataset.user_id != get_jwt_identity():
            abort(403, message='Dataset with id={} can be accessed only by an owner'.format(dataset_id))

        status = self.synthesizer_manager.get_status(dataset_id)
        if status == ModelStatus.NO_MODEL:
            abort(404, message="No training for dataset_id " + dataset_id)
        if status == ModelStatus.TRAINING:
            return {'status': 'training'}, 200
        if status == ModelStatus.FAILED:
            return {'status': 'failed'}, 200

        return {'status': 'ready'}, 200

    def post(self, dataset_id):
        dataset = self.dataset_repo.get(dataset_id)
        app.logger.info('dataset by id={} is {}'.format(dataset_id, dataset))
        if not dataset:
            abort(404, message="Couldn't find requested dataset: " + dataset_id)
        if dataset.user_id != get_jwt_identity():
            abort(403, message='Dataset with id={} can be accessed only by an owner'.format(dataset_id))

        self.synthesizer_manager.train_async(dataset_id)

        return {'status': 'training'}, 202, {'Location': '/datasets/{}/model'.format(dataset_id)}


class SynthesisResource(Resource):
    decorators = [jwt_required]

    def __init__(self, **kwargs):
        self.dataset_repo = kwargs['dataset_repo']
        self.synthesis_repo = kwargs['synthesis_repo']

    def get(self, dataset_id):
        parser = reqparse.RequestParser()
        parser.add_argument('sample_size', type=int, default=SAMPLE_SIZE)
        args = parser.parse_args()

        sample_size = args['sample_size']
        if sample_size > MAX_SAMPLE_SIZE:
            abort(400, message='Sample size is too big: ' + str(sample_size))

        dataset = self.dataset_repo.get(dataset_id)
        app.logger.info('dataset by id={} is {}'.format(dataset_id, dataset))
        if not dataset:
            abort(404, message="Couldn't find requested dataset: " + dataset_id)
        if dataset.user_id != get_jwt_identity():
            abort(403, message='Dataset with id={} can be accessed only by an owner'.format(dataset_id))

        syntheses = self.synthesis_repo.find_by_props({'dataset_id': dataset_id})
        app.logger.info('synthesis by dataset_id={} is {}'.format(dataset_id, syntheses))

        if len(syntheses) == 0:
            abort(404, messsage="Couldn't find requested synthesis")

        synthesis = syntheses[0]  # assumes the only one synthesis always

        data = pd.read_csv(BytesIO(synthesis.blob))

        return jsonify({
            'meta': simplejson.load(BytesIO(synthesis.meta), encoding='utf-8'),
            'sample': data[:sample_size].to_dict(orient='list'),
            'size': synthesis.size
        })

    def post(self, dataset_id):
        parser = reqparse.RequestParser()
        parser.add_argument('rows', type=int, required=True)
        args = parser.parse_args()

        rows = args['rows']

        app.logger.info('synthesis for dataset_id={}'.format(dataset_id))

        dataset = self.dataset_repo.get(dataset_id)
        app.logger.info('dataset by id={} is {}'.format(dataset_id, dataset))
        if not dataset:
            abort(404, message="Couldn't find requested dataset: " + dataset_id)
        if dataset.user_id != get_jwt_identity():
            abort(403, message='Dataset with id={} can be accessed only by an owner'.format(dataset_id))

        model = synthesizer_manager.get_model(dataset_id)
        app.logger.info('found a model {}'.format(model))
        if not model:
            abort(409, message='Model does not exist for user={} and dataset={}'.format(get_jwt_identity(), dataset_id))

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

        old_syntheses = self.synthesis_repo.find_by_props({'dataset_id': dataset_id})
        for s in old_syntheses:
            app.logger.info('deleting an old synthesis: {}'.format(s))
            self.synthesis_repo.delete(s)

        self.synthesis_repo.save(synthesis)

        app.logger.info('created a synthesis {}'.format(synthesis))

        return '', 204, {'Location': '/datasets/{}/synthesis'.format(dataset_id)}


class StatusResource(Resource):
    def get(self):
        return {'success': True}


dataset_repo = SQLAlchemyRepository(db, Dataset)
synthesis_repo = SQLAlchemyRepository(db, Synthesis)
user_repo = SQLAlchemyRepository(db, User)
authenticator = Authenticator(user_repo)

# each model is about 275MB in RAM
synthesizer_manager = SynthesizerManager(dataset_repo=dataset_repo, max_models=15)

api = Api(app)
api.add_resource(StatusResource, '/')
api.add_resource(LoginResource, '/login', resource_class_kwargs={'authenticator': authenticator})
api.add_resource(RefreshResource, '/refresh')
api.add_resource(UsersResource, '/users', resource_class_kwargs={'user_repo': user_repo})
api.add_resource(DatasetsResource, '/datasets', resource_class_kwargs={'dataset_repo': dataset_repo})
api.add_resource(DatasetResource, '/datasets/<dataset_id>', resource_class_kwargs={'dataset_repo': dataset_repo})
api.add_resource(DatasetUpdateInfoResource, '/datasets/<dataset_id>/updateinfo', resource_class_kwargs={'dataset_repo': dataset_repo})
api.add_resource(ModelResource, '/datasets/<dataset_id>/model', resource_class_kwargs={'dataset_repo': dataset_repo, 'synthesizer_manager': synthesizer_manager})
api.add_resource(SynthesisResource, '/datasets/<dataset_id>/synthesis', resource_class_kwargs={'dataset_repo': dataset_repo, 'synthesis_repo': synthesis_repo})
