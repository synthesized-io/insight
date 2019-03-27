from collections import namedtuple
from io import BytesIO
from operator import attrgetter
from typing import Iterable

import pandas as pd
import simplejson
from flask import current_app, jsonify
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restful import Resource, reqparse, abort
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

from .common import DatasetAccessMixin
from ..domain.correlation import compute_correlation_similarity
from ..domain.dataset_meta import DatasetMeta, DENSITY_PLOT_TYPE
from ..domain.model import Report, ReportItem, ReportItemType, Dataset
from ..domain.modelling import r2_regression_score, roc_auc_classification_score
from ..domain.repository import Repository

REGRESSORS = {
    'LinearRegression': LinearRegression,
    'GradientBoostingRegressor': GradientBoostingRegressor
}

CLASSIFIERS = {
    'LogisticRegression': LogisticRegression,
    'GradientBoostingClassifier': GradientBoostingClassifier
}

# Reports API
#
# GET  /datasets/123/report
# POST /datasets/123/report-items
# GET  /datasets/123/report-items/123
# POST /datasets/123/reports-items/123/move
# POST /datasets/123/reports-items/123/updatesettings

DEFAULT_MAX_SAMPLE_SIZE = 1000


class ReportResource(Resource, DatasetAccessMixin):
    decorators = [jwt_required]

    def __init__(self, **kwargs):
        self.dataset_repo: Repository = kwargs['dataset_repo']
        self.report_repo: Repository = kwargs['report_repo']

    def get(self, dataset_id):
        dataset: Dataset = self.get_dataset_authorized(dataset_id)
        # Parse JSON into an object with attributes corresponding to dict keys.
        meta: DatasetMeta = simplejson.load(BytesIO(dataset.meta), encoding='utf-8', object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

        reports = self.report_repo.find_by_props({'dataset_id': dataset_id})
        if len(reports) > 0:
            report = reports[0]
        else:
            return {}, 200

        items: Iterable[ReportItem] = sorted(report.items, key=attrgetter('ord'))

        item_views = []
        for report_item in items:
            if report_item.settings:
                settings = simplejson.load(BytesIO(report_item.settings), encoding='utf-8')
            else:
                settings = {}
            if report_item.results:
                results = simplejson.load(BytesIO(report_item.results), encoding='utf-8')
            else:
                results = {}
            if report_item.item_type == ReportItemType.CORRELATION:
                columns = []
                for column_meta in meta.columns:
                    if column_meta.plot_type == DENSITY_PLOT_TYPE:  # TODO: replace with type
                        columns.append(column_meta.name)
                item_views.append({
                    'id': report_item.id,
                    'type': report_item.item_type.name,
                    'order': report_item.ord,
                    'results': results,
                    'options': {
                        'columns': columns
                    },
                    'settings': settings
                })
            elif report_item.item_type == ReportItemType.MODELLING:
                continuous_columns = []
                categorical_columns = []
                for column_meta in meta.columns:
                    if column_meta.plot_type == DENSITY_PLOT_TYPE:  # TODO: replace with type
                        continuous_columns.append(column_meta.name)
                    else:
                        categorical_columns.append(column_meta.name)
                item_views.append({
                    'id': report_item.id,
                    'order': report_item.ord,
                    'results': results,
                    'options': {
                        'continuous_columns': continuous_columns,
                        'categorical_columns': categorical_columns,
                        'continuous_models': list(REGRESSORS.keys()),
                        'categorical_models': list(CLASSIFIERS.keys()),
                    },
                    'settings': settings
                })
            else:
                abort(409, message='Unknown item type: ' + str(report_item.item_type))

        return jsonify({
            'items': item_views
        })


class ReportItemsResource(Resource):
    decorators = [jwt_required]

    def __init__(self, **kwargs):
        self.dataset_repo: Repository = kwargs['dataset_repo']
        self.report_repo: Repository = kwargs['report_repo']
        self.report_item_repo: Repository = kwargs['report_item_repo']

    def post(self, dataset_id):
        parser = reqparse.RequestParser()
        parser.add_argument('type', type=str, choices=('CORRELATION', 'MODELLING'), required=True)
        args = parser.parse_args()

        type = args['type']

        dataset = self.dataset_repo.get(dataset_id)
        current_app.logger.info('dataset by id={} is {}'.format(dataset_id, dataset))
        if not dataset:
            abort(404, message="Couldn't find requested dataset: " + dataset_id)
        if dataset.user_id != get_jwt_identity():
            abort(403, message='Dataset with id={} can be accessed only by an owner'.format(dataset_id))

        reports = self.report_repo.find_by_props({'dataset_id': dataset_id})
        if len(reports) == 0:
            report = Report(dataset_id=dataset_id)
            self.report_repo.save(report)
        elif len(reports) > 1:
            abort(409, message='More than one report has been found')
        else:
            report = reports[0]

        exisitng_items = report.items
        if len(exisitng_items) > 0:
            ord = max(map(attrgetter('ord'), exisitng_items)) + 1
        else:
            ord = 0

        report_item = ReportItem(ord=ord, report_id=report.id, item_type=type)
        self.report_item_repo.save(report_item)

        return {'id': report_item.id}, 201, {'Location': '/datasets/{}/report-items/{}'.format(dataset_id, report_item.id)}


class ReportItemsMoveResource(Resource):
    decorators = [jwt_required]

    def post(self, dataset_id, report_item_id):
        pass


class ReportItemsUpdateSettingsResource(Resource, DatasetAccessMixin):
    decorators = [jwt_required]

    def __init__(self, **kwargs):
        self.dataset_repo: Repository = kwargs['dataset_repo']
        self.report_item_repo: Repository = kwargs['report_item_repo']
        self.synthesis_repo: Repository = kwargs['synthesis_repo']

    def post(self, dataset_id, report_item_id):
        dataset = self.get_dataset_authorized(dataset_id)

        report_item: ReportItem = self.report_item_repo.get(report_item_id)
        if not report_item:
            abort(404, message='Requested item has not been found: ' + str(report_item_id))

        parser = reqparse.RequestParser()
        parser.add_argument('settings', type=dict, required=True)
        parser.add_argument('max_sample_size', type=int, default=DEFAULT_MAX_SAMPLE_SIZE)
        args = parser.parse_args()

        settings = args['settings']
        max_sample_size = args['max_sample_size']

        syntheses = self.synthesis_repo.find_by_props({'dataset_id': dataset_id})
        if len(syntheses) == 0:
            abort(409, message='Not synthesis found for dataset: ' + str(dataset_id))
        elif len(syntheses) > 1:
            abort(409, message='There is more than one synthesis for dataset: ' + str(dataset_id))
        else:
            synthesis = syntheses[0]

        df_orig = pd.read_csv(BytesIO(dataset.blob), encoding='utf-8')
        df_synth = pd.read_csv(BytesIO(synthesis.blob), encoding='utf-8')

        if report_item.item_type == ReportItemType.CORRELATION:
            columns = settings['columns']
            correlation_similarity = compute_correlation_similarity(df_orig, df_synth, columns)

            size = min(len(df_orig), len(df_synth), max_sample_size)
            df_orig_sample = df_orig[columns].dropna().sample(size)
            df_synth_sample = df_synth[columns].dropna().sample(size)

            results = {
                'correlation_similarity': correlation_similarity.to_dict(),
                'original_sample': df_orig_sample.to_dict(orient='list'),
                'synthetic_sample': df_synth_sample.to_dict(orient='list')
            }
            report_item.results = simplejson.dumps(results).encode('utf-8')
        elif report_item.item_type == ReportItemType.MODELLING:
            response_variable = settings['response_variable']
            explanatory_variables = settings['explanatory_variables']
            model = settings['model']

            df_train, df_test = train_test_split(df_orig, test_size=0.2, random_state=42)

            meta = simplejson.load(BytesIO(dataset.meta), encoding='utf-8', object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

            results = {}
            if model in REGRESSORS:
                model_class = REGRESSORS[model]
                try:
                    orig_score = r2_regression_score(model_class(), df_train, df_test, meta, explanatory_variables, response_variable)
                    synth_score = r2_regression_score(model_class(), df_synth, df_test, meta, explanatory_variables, response_variable)
                    results['metric'] = 'r2'
                    results['original_score'] = orig_score
                    results['synthetic_score'] = synth_score
                except Exception as e:
                    current_app.logger.error(e)
                    results['error'] = str(e)
            elif model in CLASSIFIERS:
                model_class = CLASSIFIERS[model]
                try:
                    orig_score = roc_auc_classification_score(model_class(), df_train, df_test, meta, explanatory_variables, response_variable)
                    synth_score = roc_auc_classification_score(model_class(), df_synth, df_test, meta, explanatory_variables, response_variable)
                    results['metric'] = 'roc_auc'
                    results['original_score'] = orig_score
                    results['synthetic_score'] = synth_score
                except Exception as e:
                    current_app.logger.error(e)
                    results['error'] = str(e)
            else:
                abort(400, message='Unknown model: ' + model)

            report_item.results = simplejson.dumps(results).encode('utf-8')

        settings_blob = simplejson.dumps(settings).encode('utf-8')
        report_item.settings = settings_blob
        self.report_item_repo.save(report_item)

        return '', 204
