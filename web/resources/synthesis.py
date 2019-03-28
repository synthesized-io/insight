from io import StringIO, BytesIO

import pandas as pd
import simplejson
from flask import current_app
from flask import jsonify
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restful import Resource, reqparse, abort

from ..application.synthesizer_manager import ModelStatus
from ..application.synthesizer_manager import SynthesizerManager
from ..domain.dataset_meta import recompute_dataset_meta
from ..domain.model import Synthesis
from ..domain.repository import Repository

SAMPLE_SIZE = 20
MAX_SAMPLE_SIZE = 10000


class ModelResource(Resource):
    decorators = [jwt_required]

    def __init__(self, **kwargs):
        self.dataset_repo: Repository = kwargs['dataset_repo']
        self.synthesizer_manager: SynthesizerManager = kwargs['synthesizer_manager']

    def get(self, dataset_id):
        dataset = self.dataset_repo.get(dataset_id)
        current_app.logger.info('dataset by id={} is {}'.format(dataset_id, dataset))
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
        current_app.logger.info('dataset by id={} is {}'.format(dataset_id, dataset))
        if not dataset:
            abort(404, message="Couldn't find requested dataset: " + dataset_id)
        if dataset.user_id != get_jwt_identity():
            abort(403, message='Dataset with id={} can be accessed only by an owner'.format(dataset_id))

        self.synthesizer_manager.train_async(dataset_id)

        return {'status': 'training'}, 202, {'Location': '/datasets/{}/model'.format(dataset_id)}


class SynthesisResource(Resource):
    decorators = [jwt_required]

    def __init__(self, **kwargs):
        self.dataset_repo: Repository = kwargs['dataset_repo']
        self.synthesis_repo: Repository = kwargs['synthesis_repo']
        self.synthesizer_manager: SynthesizerManager = kwargs['synthesizer_manager']

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
            abort(404, message="Couldn't find requested dataset: " + dataset_id)
        if dataset.user_id != get_jwt_identity():
            abort(403, message='Dataset with id={} can be accessed only by an owner'.format(dataset_id))

        syntheses = self.synthesis_repo.find_by_props({'dataset_id': dataset_id})
        current_app.logger.info('synthesis by dataset_id={} is {}'.format(dataset_id, syntheses))

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

        current_app.logger.info('synthesis for dataset_id={}'.format(dataset_id))

        dataset = self.dataset_repo.get(dataset_id)
        current_app.logger.info('dataset by id={} is {}'.format(dataset_id, dataset))
        if not dataset:
            abort(404, message="Couldn't find requested dataset: " + dataset_id)
        if dataset.user_id != get_jwt_identity():
            abort(403, message='Dataset with id={} can be accessed only by an owner'.format(dataset_id))

        model = self.synthesizer_manager.get_model(dataset_id)
        current_app.logger.info('found a model {}'.format(model))
        if not model:
            abort(409, message='Model does not exist for user={} and dataset={}'.format(get_jwt_identity(), dataset_id))

        current_app.logger.info('starting synthesis')
        synthesized = model.synthesize(rows)

        output = StringIO()
        synthesized.to_csv(output, index=False, encoding='utf-8')

        blob = output.getvalue().encode('utf-8')

        original_meta = dataset.get_meta_as_object()
        synthetic_meta = recompute_dataset_meta(synthesized, original_meta)

        synthesis = Synthesis(dataset_id=dataset_id, blob=blob, size=rows)
        synthesis.set_meta_from_object(synthetic_meta)

        old_syntheses = self.synthesis_repo.find_by_props({'dataset_id': dataset_id})
        for s in old_syntheses:
            current_app.logger.info('deleting an old synthesis: {}'.format(s))
            self.synthesis_repo.delete(s)

        self.synthesis_repo.save(synthesis)

        current_app.logger.info('created a synthesis {}'.format(synthesis))

        return '', 204, {'Location': '/datasets/{}/synthesis'.format(dataset_id)}
