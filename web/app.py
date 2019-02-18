import json
from io import StringIO

import numpy as np
import pandas as pd
import werkzeug
from flask import Flask
from flask_restful import Resource, Api, reqparse, abort

from synthesized.core import BasicSynthesizer
from synthesized.core.values import ContinuousValue
from .model import Dataset, Synthesis
from .repository import InMemoryRepository

SAMPLE_SIZE = 20
REMOVE_OUTLIERS = 0.01

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
api = Api(app)

datasetRepo = InMemoryRepository()
synthesisRepo = InMemoryRepository()


class DatasetsResource(Resource):
    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('file', type=werkzeug.FileStorage, location='files', required=True)
        args = parse.parse_args()
        file = args['file']

        with file.stream as stream:
            data = pd.read_csv(stream)
            output = StringIO()
            data.to_csv(output, index=False)
            synthesizer = BasicSynthesizer(data=data.dropna())
            value_types = set()
            columns_info = []
            for value in synthesizer.values:
                value_types.add(str(value))
                if isinstance(value, ContinuousValue):
                    q = [REMOVE_OUTLIERS / 2., 1 - REMOVE_OUTLIERS / 2.]
                    start, end = np.quantile(data[value.name], q)
                    column_cleaned = data[(data[value.name] > start) & (data[value.name] < end)][value.name]
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
                        'nulls': int(data[value.name].isnull().sum()),
                        'hist': hist,
                        'edges': edges
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
                        'nunique': int(data[value.name].nunique()),
                        'most_frequent': str(most_frequent),
                        'most_occurrences': int(len(data[data[value.name] == most_frequent])),
                        'hist': hist,
                        'bins': bins
                    })
            meta = {
                'rows': len(data),
                'columns': len(data.columns),
                'ntypes': len(value_types),
                'columns_info': columns_info,
                'sample': data[:SAMPLE_SIZE].to_dict(orient='list')
            }
            dataset = Dataset(None, output.getvalue(), json.dumps(meta))
            datasetRepo.save(dataset)
            return {'dataset_id': dataset.entity_id}, 201


class DatasetResource(Resource):
    def get(self, dataset_id):
        dataset = datasetRepo.find(dataset_id)
        if not dataset:
            abort(404, messsage="Couldn't find requested dataset: " + dataset_id)
        return {
            'dataset_id': dataset.entity_id,
            'meta': json.loads(dataset.meta)
        }

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

        synthesis = Synthesis(None, dataset_id, output.getvalue(), rows)
        synthesisRepo.save(synthesis)

        return {'synthesis_id': synthesis.entity_id}, 201


class SynthesisResource(Resource):
    def get(self, synthesis_id):
        synthesis = synthesisRepo.find(synthesis_id)
        if not synthesis:
            abort(404, messsage="Couldn't find requested synthesis: " + synthesis_id)

        data = pd.read_csv(StringIO(synthesis.blob))

        return {
            'synthesis_id': synthesis_id,
            'meta': {
                'sample': data[:SAMPLE_SIZE].to_dict(orient='list')
            }
        }


api.add_resource(DatasetsResource, '/dataset')
api.add_resource(DatasetResource, '/dataset/<dataset_id>')
api.add_resource(SynthesesResource, '/synthesis')
api.add_resource(SynthesisResource, '/synthesis/<synthesis_id>')
