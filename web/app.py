from io import StringIO
from threading import Lock

import numpy as np
import pandas as pd
import simplejson
import werkzeug
from flask import Flask, jsonify
from flask.json import JSONEncoder
from flask_bcrypt import Bcrypt
from flask_jwt import JWT, jwt_required, current_identity
from flask_restful import Resource, Api, reqparse, abort
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS

from synthesized.core import BasicSynthesizer
from synthesized.core.values import ContinuousValue
from .config import ProductionConfig, DevelopmentConfig
from .repository import SQLAlchemyRepository
import os

SAMPLE_SIZE = 20
MAX_SAMPLE_SIZE = 10000
REMOVE_OUTLIERS = 0.01


# By default NaN is serialized as "NaN". We enforce "null" instead.
class JSONCompliantEncoder(JSONEncoder):
    def __init__(self, *args, **kwargs):
        kwargs["ignore_nan"] = True
        super().__init__(*args, **kwargs)


app = Flask(__name__)
CORS(app)
app.json_encoder = JSONCompliantEncoder
app.config['SQLALCHEMY_ECHO'] = False
app.config['PROPAGATE_EXCEPTIONS'] = True  # this is crucial to enable handling of JWTError
if app.config['ENV'] == 'production':
    app.config.from_object(ProductionConfig)
else:
    app.config.from_object(DevelopmentConfig)
app.config['SECRET_KEY'] = 'synthesized-secret'

db = SQLAlchemy(app)
# models use `db` object, therefore should be imported after
from .model import Dataset, Synthesis, User
db.create_all()

datasetRepo = SQLAlchemyRepository(db, Dataset)
synthesisRepo = SQLAlchemyRepository(db, Synthesis)
userRepo = SQLAlchemyRepository(db, User)

bcrypt = Bcrypt(app)


def authenticate(username, password):
    users = userRepo.find_by_props({'username': username})
    if len(users) > 0:
        password_hash = bytes.fromhex(users[0].password)
        if bcrypt.check_password_hash(password_hash, password):
            return users[0]


def identity(payload):
    user_id = payload['identity']
    return userRepo.get(user_id)


jwt = JWT(app, authenticate, identity)


class UsersResource(Resource):
    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('username', type=str, required=True)
        parse.add_argument('password', type=str, required=True)
        args = parse.parse_args()

        username = args['username']
        password = args['password']

        app.logger.info('registering user {}'.format(username))

        users = userRepo.find_by_props({'username': username})
        if len(users) > 0:
            app.logger.info('found existing user {}'.format(users[0]))
            abort(409, message='User with username={} already exists'.format(username))

        user = User(username=username, password=bcrypt.generate_password_hash(password).hex())
        userRepo.save(user)
        app.logger.info('created a user {}'.format(user))

        return {'user_id': user.id}, 201


class DatasetsResource(Resource):
    decorators = [jwt_required()]

    def get(self):
        datasets = datasetRepo.find_by_props({})
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
        parse = reqparse.RequestParser()
        parse.add_argument('file', type=werkzeug.FileStorage, location='files', required=True)
        args = parse.parse_args()
        file = args['file']

        app.logger.info('processing file {}'.format(file.filename))

        with file.stream as stream:
            data = pd.read_csv(stream)
            raw_data = StringIO()
            data.to_csv(raw_data, index=False)
            data_wo_nans = data.dropna()
            synthesizer = BasicSynthesizer(data=data_wo_nans)
            value_types = set()
            columns_info = []
            for value in synthesizer.values:
                value_types.add(str(value))
                if isinstance(value, ContinuousValue):
                    q = [REMOVE_OUTLIERS / 2., 1 - REMOVE_OUTLIERS / 2.]
                    start, end = np.quantile(data_wo_nans[value.name], q)
                    column_cleaned = data_wo_nans[(data_wo_nans[value.name] > start) & (data_wo_nans[value.name] < end)][value.name]
                    hist, edges = np.histogram(column_cleaned, bins='auto')
                    hist = list(map(int, hist))
                    edges = list(map(float, edges))
                    columns_info.append({
                        'name': value.name,
                        'plot_type': 'density',
                        'type': str(value),
                        'mean': float(data[value.name].mean()),
                        'std': float(data[value.name].std()),
                        'median': float(data[value.name].median()),
                        'min': float(data[value.name].min()),
                        'max': float(data[value.name].max()),
                        'n_nulls': int(data[value.name].isnull().sum()),
                        'plot_data': {
                            'hist': hist,
                            'edges': edges
                        }
                    })
                else:
                    most_frequent = data[value.name].value_counts().idxmax()
                    bins = sorted(data[value.name].dropna().unique())
                    counts = data[value.name].value_counts().to_dict()
                    hist = [counts[x] for x in bins]
                    bins = list(map(str, bins))
                    hist = list(map(int, hist))
                    columns_info.append({
                        'name': value.name,
                        'plot_type': 'histogram',
                        'type': str(value),
                        'n_unique': int(data[value.name].nunique()),
                        'most_frequent': str(most_frequent),
                        'most_occurrences': int(len(data[data[value.name] == most_frequent])),
                        'plot_data': {
                            'hist': hist,
                            'bins': bins
                        }
                    })
            meta = {
                'n_rows': len(data),
                'n_columns': len(data.columns),
                'n_types': len(value_types),
                'columns': columns_info,
            }
            title = os.path.splitext(file.filename)[0]
            dataset = Dataset(user_id=current_identity.id, title=title, blob=raw_data.getvalue(), meta=simplejson.dumps(meta, ignore_nan=True))
            datasetRepo.save(dataset)
            app.logger.info('created a dataset {}'.format(dataset))

            return {'dataset_id': dataset.id}, 201


class DatasetResource(Resource):
    decorators = [jwt_required()]

    def get(self, dataset_id):
        parse = reqparse.RequestParser()
        parse.add_argument('sample_size', type=int, default=SAMPLE_SIZE)
        args = parse.parse_args()

        sample_size = args['sample_size']
        if sample_size > MAX_SAMPLE_SIZE:
            abort(400, message='Sample size is too big: ' + str(sample_size))

        dataset = datasetRepo.get(dataset_id)
        app.logger.info('dataset by id={} is {}'.format(dataset_id, dataset))

        if not dataset:
            abort(404, messsage="Couldn't find requested dataset: " + dataset_id)
        if dataset.user_id != current_identity.id:
            abort(403, message='Dataset with id={} can be accessed only by an owner'.format(dataset_id))

        data = pd.read_csv(StringIO(dataset.blob))

        return jsonify({
            'dataset_id': dataset.id,
            'title': dataset.title,
            'description': dataset.description,
            'meta': simplejson.loads(dataset.meta),
            'sample': data[:sample_size].to_dict(orient='list')
        })

    def delete(self, dataset_id):
        dataset = datasetRepo.get(dataset_id)
        app.logger.info('dataset by id={} is {}'.format(dataset_id, dataset))
        if dataset:
            if dataset.user_id != current_identity.id:
                abort(403, message='Dataset with id={} can be deleted only by an owner'.format(dataset_id))
            datasetRepo.delete(dataset)
        return '', 204


class DatasetUpdateInfoResource(Resource):
    decorators = [jwt_required()]

    def post(self, dataset_id):
        parse = reqparse.RequestParser()
        parse.add_argument('title', type=str, required=False)
        parse.add_argument('description', type=str, required=False)
        args = parse.parse_args()

        dataset = datasetRepo.get(dataset_id)
        app.logger.info('dataset by id={} is {}'.format(dataset_id, dataset))
        if not dataset:
            abort(404, messsage="Couldn't find requested dataset: " + dataset_id)

        app.logger.info('updating the dataset with args {}'.format(args))

        dataset.title = args['title']
        dataset.description = args['description']

        db.session.commit()

        return '', 204


models_lock = Lock()
models = {}


class ModelResource(Resource):
    decorators = [jwt_required()]

    def get(self, dataset_id):
        model_key = (current_identity.id, dataset_id)
        with models_lock:
            model = models.get(model_key, None)
            if not model:
                abort(404, message='Model does not exist for user={} and dataset={}'.format(current_identity.id, dataset_id))
        return {'model': str(model)}, 200

    def post(self, dataset_id):
        dataset = datasetRepo.get(dataset_id)
        app.logger.info('dataset by id={} is {}'.format(dataset_id, dataset))
        if not dataset:
            abort(404, message="Couldn't find requested dataset: " + dataset_id)
        if dataset.user_id != current_identity.id:
            abort(403, message='Dataset with id={} can be accessed only by an owner'.format(dataset_id))

        model_key = (current_identity.id, dataset_id)
        with models_lock:
            model = models.get(model_key, None)
            app.logger.info('model by key={} is {}'.format(model_key, model))
            if model:
                return '', 204

        data = pd.read_csv(StringIO(dataset.blob))
        data = data.dropna()

        app.logger.info('starting model training')
        synthesizer = BasicSynthesizer(data=data)
        synthesizer.__enter__()
        synthesizer.learn(data=data)
        app.logger.info('model has been trained')

        with models_lock:
            models[model_key] = synthesizer

        return '', 204


class SynthesesResource(Resource):
    decorators = [jwt_required()]

    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('dataset_id', type=str, required=True)
        parse.add_argument('rows', type=int, required=True)
        args = parse.parse_args()

        dataset_id = args['dataset_id']
        rows = args['rows']

        app.logger.info('synthesis for dataset_id={}'.format(dataset_id))

        model_key = (current_identity.id, dataset_id)
        app.logger.info('model key is {}'.format(model_key))
        with models_lock:
            model = models.get(model_key, None)
            app.logger.info('found a model {}'.format(model))
            if not model:
                abort(404, message='Model does not exist for user={} and dataset={}'.format(current_identity.id, dataset_id))

        app.logger.info('starting synthesis')
        synthesized = model.synthesize(rows)

        output = StringIO()
        synthesized.to_csv(output, index=False)

        synthesis = Synthesis(dataset_id=dataset_id, blob=output.getvalue(), size=rows)
        synthesisRepo.save(synthesis)

        app.logger.info('created a synthesis {}'.format(synthesis))

        return {'synthesis_id': synthesis.id}, 201


class SynthesisResource(Resource):
    decorators = [jwt_required()]

    def get(self, synthesis_id):
        parse = reqparse.RequestParser()
        parse.add_argument('sample_size', type=int, default=SAMPLE_SIZE)
        args = parse.parse_args()

        sample_size = args['sample_size']
        if sample_size > MAX_SAMPLE_SIZE:
            abort(400, message='Sample size is too big: ' + str(sample_size))

        synthesis = synthesisRepo.get(synthesis_id)
        app.logger.info('synthesis by id={} is {}'.format(synthesis_id, synthesis))

        if not synthesis:
            abort(404, messsage="Couldn't find requested synthesis: " + synthesis_id)

        data = pd.read_csv(StringIO(synthesis.blob))

        return jsonify({
            'synthesis_id': synthesis_id,
            'meta': {
            },
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
api.add_resource(SynthesesResource, '/syntheses')
api.add_resource(SynthesisResource, '/syntheses/<synthesis_id>')
