from io import BytesIO
from operator import attrgetter
from typing import Iterable

import pandas as pd
import simplejson
from flask import current_app, jsonify
from flask_jwt_extended import jwt_required
from flask_restful import Resource, reqparse, abort
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split

from .common import DatasetAccessMixin
from ..application.report_item_ordering import ReportItemOrdering
from ..domain import dataset_meta
from ..domain.correlation import compute_correlation_similarity
from ..domain.dataset_meta import DatasetMeta
from ..domain.model import Report, ReportItem, ReportItemType, Dataset
from ..domain.modelling import r2_regression_score, roc_auc_classification_score
from ..domain.quality import quality_pct
from ..domain.repository import Repository

REGRESSORS = {
    'LinearRegression': LinearRegression,
    'GradientBoostingRegressor': GradientBoostingRegressor
}

CLASSIFIERS = {
    'LogisticRegression': LogisticRegression,
    'GradientBoostingClassifier': GradientBoostingClassifier
}

DEFAULT_MAX_SAMPLE_SIZE = 100


class ReportResource(Resource, DatasetAccessMixin):
    decorators = [jwt_required]

    def __init__(self, **kwargs):
        self.dataset_repo: Repository = kwargs['dataset_repo']
        self.report_repo: Repository = kwargs['report_repo']

    def get(self, dataset_id):
        dataset: Dataset = self.get_dataset_authorized(dataset_id)
        meta: DatasetMeta = dataset.get_meta_as_object()
        disabled_columns = dataset.get_settings_as_dict().get('disabled_columns', [])

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
                    if column_meta.name in disabled_columns:
                        continue
                    if column_meta.type_family == dataset_meta.CONTINUOUS_TYPE_FAMILY:
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
                    if column_meta.name in disabled_columns:
                        continue
                    if column_meta.type_family == dataset_meta.CONTINUOUS_TYPE_FAMILY:
                        continuous_columns.append(column_meta.name)
                    elif column_meta.type_family == dataset_meta.CATEGORICAL_TYPE_FAMILY:
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


class ReportItemsResource(Resource, DatasetAccessMixin):
    decorators = [jwt_required]

    def __init__(self, **kwargs):
        self.dataset_repo: Repository = kwargs['dataset_repo']
        self.report_repo: Repository = kwargs['report_repo']
        self.report_item_repo: Repository = kwargs['report_item_repo']

    def post(self, dataset_id):
        self.get_dataset_authorized(dataset_id)

        parser = reqparse.RequestParser()
        parser.add_argument('type', type=str, choices=('CORRELATION', 'MODELLING'), required=True)
        args = parser.parse_args()

        type = args['type']

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


class ReportItemResource(Resource, DatasetAccessMixin):
    decorators = [jwt_required]

    def __init__(self, **kwargs):
        self.dataset_repo: Repository = kwargs['dataset_repo']
        self.report_repo: Repository = kwargs['report_repo']
        self.report_item_repo: Repository = kwargs['report_item_repo']

    def delete(self, dataset_id, report_item_id):
        self.get_dataset_authorized(dataset_id)

        report_item: ReportItem = self.report_item_repo.get(report_item_id)
        if report_item:
            report: Report = self.report_repo.get(report_item.report_id)
            if report.dataset_id != int(dataset_id):
                abort(403, message='Requested item does not correspond to the given dataset')

            self.report_item_repo.delete(report_item)

        return '', 204


class ReportItemsUpdateSettingsResource(Resource, DatasetAccessMixin):
    decorators = [jwt_required]

    def __init__(self, **kwargs):
        self.dataset_repo: Repository = kwargs['dataset_repo']
        self.report_repo: Repository = kwargs['report_repo']
        self.report_item_repo: Repository = kwargs['report_item_repo']
        self.synthesis_repo: Repository = kwargs['synthesis_repo']

    def post(self, dataset_id, report_item_id):
        dataset = self.get_dataset_authorized(dataset_id)

        report_item: ReportItem = self.report_item_repo.get(report_item_id)
        if not report_item:
            abort(404, message='Requested item has not been found: ' + str(report_item_id))

        report: Report = self.report_repo.get(report_item.report_id)
        if report.dataset_id != int(dataset_id):
            abort(403, message='Requested item does not correspond to the given dataset')

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

            meta = dataset.get_meta_as_object()

            results = {}
            if model in REGRESSORS:
                model_class = REGRESSORS[model]
                try:
                    orig_score = r2_regression_score(model_class(), df_train, df_test, meta, explanatory_variables, response_variable)
                    synth_score = r2_regression_score(model_class(), df_synth, df_test, meta, explanatory_variables, response_variable)
                    results['metric'] = 'r2'
                    results['original_score'] = orig_score
                    results['synthetic_score'] = synth_score
                    results['quality'] = quality_pct(orig_score, synth_score)
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
                    results['quality'] = quality_pct(orig_score, synth_score)
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


class ReportItemsMoveResource(Resource, DatasetAccessMixin):
    decorators = [jwt_required]

    def __init__(self, **kwargs):
        self.dataset_repo: Repository = kwargs['dataset_repo']
        self.report_repo: Repository = kwargs['report_repo']
        self.report_item_repo: Repository = kwargs['report_item_repo']
        self.report_item_ordering: ReportItemOrdering = kwargs['report_item_ordering']

    def post(self, dataset_id, report_item_id):
        self.get_dataset_authorized(dataset_id)

        report_item: ReportItem = self.report_item_repo.get(report_item_id)
        if not report_item:
            abort(404, message='Requested item has not been found: ' + str(report_item_id))

        report: Report = self.report_repo.get(report_item.report_id)
        if report.dataset_id != int(dataset_id):
            abort(403, message='Requested item does not correspond to the given dataset')

        parser = reqparse.RequestParser()
        parser.add_argument('new_order', type=int, required=True)
        args = parser.parse_args()

        new_order = args['new_order']

        self.report_item_ordering.move_item(report_item_id, new_order)

        return '', 204
