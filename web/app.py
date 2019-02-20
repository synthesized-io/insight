import logging
from io import StringIO

import numpy as np
import pandas as pd
import simplejson
import werkzeug
from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse, abort
from flask_sqlalchemy import SQLAlchemy
from flask.json import JSONEncoder

from synthesized.core import BasicSynthesizer
from synthesized.core.values import ContinuousValue
from .repository import SQLAlchemyRepository
from .config import ProductionConfig, DevelopmentConfig

SAMPLE_SIZE = 20
MAX_SAMPLE_SIZE = 10000
REMOVE_OUTLIERS = 0.01

logging.basicConfig(level=logging.INFO)


# By default NaN is serialized as "NaN". We enforce "null" instead.
class JSONCompliantEncoder(JSONEncoder):
    def __init__(self, *args, **kwargs):
        kwargs["ignore_nan"] = True
        super().__init__(*args, **kwargs)


app = Flask(__name__)
app.json_encoder = JSONCompliantEncoder
if app.config['ENV'] == 'production':
    app.config.from_object(ProductionConfig)
else:
    app.config.from_object(DevelopmentConfig)

api = Api(app)

db = SQLAlchemy(app)
# models use `db` object, therefore should be imported after
from .model import Dataset, Synthesis
db.create_all()

datasetRepo = SQLAlchemyRepository(db, Dataset)
synthesisRepo = SQLAlchemyRepository(db, Synthesis)


class DatasetsResource(Resource):
    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('file', type=werkzeug.FileStorage, location='files', required=True)
        args = parse.parse_args()
        file = args['file']

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
            dataset = Dataset(blob=raw_data.getvalue(), meta=simplejson.dumps(meta, ignore_nan=True))
            datasetRepo.save(dataset)
            return {'dataset_id': dataset.id}, 201


class DatasetResource(Resource):
    def get(self, dataset_id):
        parse = reqparse.RequestParser()
        parse.add_argument('sample_size', type=int, default=SAMPLE_SIZE)
        args = parse.parse_args()

        sample_size = args['sample_size']
        if sample_size > MAX_SAMPLE_SIZE:
            abort(400, message='Sample size is too big: ' + str(sample_size))

        dataset = datasetRepo.find(dataset_id)
        if not dataset:
            abort(404, messsage="Couldn't find requested dataset: " + dataset_id)

        data = pd.read_csv(StringIO(dataset.blob))

        return jsonify({
            'dataset_id': dataset.id,
            'meta': simplejson.loads(dataset.meta),
            'sample': data[:sample_size].to_dict(orient='list')
        })

    def delete(self, dataset_id):
        datasetRepo.delete(dataset_id)
        return '', 204


class SynthesesResource(Resource):
    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('dataset_id', type=str, required=True)
        parse.add_argument('rows', type=int, required=True)
        args = parse.parse_args()

        dataset_id = args['dataset_id']
        rows = args['rows']

        dataset = datasetRepo.find(dataset_id)
        if not dataset:
            abort(409, message="Couldn't find requested dataset: " + dataset_id)

        data = pd.read_csv(StringIO(dataset.blob))
        data = data.dropna()

        with BasicSynthesizer(data=data) as synthesizer:
            synthesizer.learn(data=data)
            synthesized = synthesizer.synthesize(rows)

        output = StringIO()
        synthesized.to_csv(output, index=False)

        synthesis = Synthesis(dataset_id=dataset_id, blob=output.getvalue(), size=rows)
        synthesisRepo.save(synthesis)

        return {'synthesis_id': synthesis.id}, 201


class SynthesisResource(Resource):
    def get(self, synthesis_id):
        parse = reqparse.RequestParser()
        parse.add_argument('sample_size', type=int, default=SAMPLE_SIZE)
        args = parse.parse_args()

        sample_size = args['sample_size']
        if sample_size > MAX_SAMPLE_SIZE:
            abort(400, message='Sample size is too big: ' + str(sample_size))

        synthesis = synthesisRepo.find(synthesis_id)
        if not synthesis:
            abort(404, messsage="Couldn't find requested synthesis: " + synthesis_id)

        data = pd.read_csv(StringIO(synthesis.blob))

        return jsonify({
            'synthesis_id': synthesis_id,
            'meta': {
            },
            'sample': data[:sample_size].to_dict(orient='list')
        })


api.add_resource(DatasetsResource, '/datasets')
api.add_resource(DatasetResource, '/datasets/<dataset_id>')
api.add_resource(SynthesesResource, '/syntheses')
api.add_resource(SynthesisResource, '/syntheses/<synthesis_id>')
