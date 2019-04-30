from web.infastructure.flask_support import configure_logger
configure_logger()

from flask import Flask, send_from_directory, safe_join, send_file
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_restful import Api as _Api
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
from flask_jwt_extended.exceptions import JWTExtendedException

from .config import ProductionConfig, DevelopmentConfig
from .infastructure.flask_support import JSONCompliantEncoder
from .infastructure.repository_impl import SQLAlchemyRepository, JsonFileDirectory
from .application.authenticator import Authenticator
import os

STATIC_DIR = 'frontend/build'
INDEX_FILE = 'index.html'

# static_folder=None disables static serving, we will use a custom one
app = Flask(__name__, static_folder=None)
if app.config['ENV'] == 'production':
    app.config.from_object(ProductionConfig)
else:
    app.config.from_object(DevelopmentConfig)
app.json_encoder = JSONCompliantEncoder


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    filename = safe_join(app.root_path, STATIC_DIR, path)
    if os.path.isfile(filename):
        cache_timeout = app.get_send_file_max_age(path)
        return send_file(filename, cache_timeout=cache_timeout)
    else:
        cache_timeout = app.get_send_file_max_age(INDEX_FILE)
        return send_from_directory(STATIC_DIR, INDEX_FILE, cache_timeout=cache_timeout)


CORS(app, supports_credentials=True)  # TODO: delete in final version

bcrypt = Bcrypt(app)

jwt = JWTManager(app)

db = SQLAlchemy(app)
# models use `db` object, therefore should be imported after `db` creation
from web.domain.model import Dataset, Synthesis, User, Report, ReportItem, UsedInvite, Entitlement
db.create_all()

dataset_repo = SQLAlchemyRepository(db, Dataset)
used_invite_repo = SQLAlchemyRepository(db, UsedInvite)
synthesis_repo = SQLAlchemyRepository(db, Synthesis)
user_repo = SQLAlchemyRepository(db, User)
report_repo = SQLAlchemyRepository(db, Report)
report_item_repo = SQLAlchemyRepository(db, ReportItem)
entitlement_repo = SQLAlchemyRepository(db, Entitlement)


authenticator = Authenticator(user_repo, bcrypt)

# should be imported after `db` creation
from .infastructure.report_item_ordering_imp import SQLAlchemyReportItemOrdering
from .infastructure.entitlement_completion_impl import SQLAlchemyEntitlementCompletion
from .application.project_templates import ProjectTemplates
from .application.synthesizer_manager import SynthesizerManager

# each model is about 275MB in RAM
synthesizer_manager = SynthesizerManager(dataset_repo=dataset_repo, max_models=15)

report_item_ordering = SQLAlchemyReportItemOrdering(db)
entitlement_completion = SQLAlchemyEntitlementCompletion(db)
template_directory = JsonFileDirectory(os.path.join(app.root_path, 'project_templates/meta.json'), 'templates')
project_templates = ProjectTemplates(template_directory, dataset_repo)

from .resources.auth import LoginResource, RefreshResource, UsersResource
from .resources.dataset import DatasetsResource, DatasetResource, DatasetUpdateInfoResource, DatasetUpdateSettingsResource, DatasetExportResource
from .resources.synthesis import ModelResource, SynthesisResource, SynthesisPreviewResource
from .resources.report import ReportItemsResource, ReportResource, ReportItemsUpdateSettingsResource, ReportItemsMoveResource, ReportItemResource
from .resources.templates import ProjectTemplatesResource, DatasetFromTemplateResource
from .resources.entitelement import EntitlementResource, EntitlementsResource, EntitlementCompletionResource


# Flask-RESTful overrides all user exception handlers.
# But we want to keep flask_jwt_extended's handlers.
class Api(_Api):
    def error_router(self, original_handler, e):
        """Based on flask_restful.Api.error_router"""
        if self._has_fr_route() and not isinstance(e, JWTExtendedException):
            try:
                return self.handle_error(e)
            except Exception:
                pass  # Fall through to original handler
        return original_handler(e)


api = Api(app, prefix='/api')
api.add_resource(LoginResource, '/login', resource_class_kwargs={'authenticator': authenticator})
api.add_resource(RefreshResource, '/refresh')
api.add_resource(UsersResource, '/users', resource_class_kwargs={'user_repo': user_repo, 'used_invite_repo': used_invite_repo, 'bcrypt': bcrypt})
api.add_resource(DatasetsResource, '/datasets', resource_class_kwargs={'dataset_repo': dataset_repo, 'entitlement_repo': entitlement_repo})
api.add_resource(DatasetResource, '/datasets/<dataset_id>', resource_class_kwargs={'dataset_repo': dataset_repo, 'entitlement_repo': entitlement_repo})
api.add_resource(DatasetUpdateInfoResource, '/datasets/<dataset_id>/updateinfo', resource_class_kwargs={'dataset_repo': dataset_repo, 'entitlement_repo': entitlement_repo})
api.add_resource(DatasetUpdateSettingsResource, '/datasets/<dataset_id>/updatesettings', resource_class_kwargs={'dataset_repo': dataset_repo, 'entitlement_repo': entitlement_repo})
api.add_resource(ModelResource, '/datasets/<dataset_id>/model', resource_class_kwargs={'dataset_repo': dataset_repo, 'synthesizer_manager': synthesizer_manager, 'entitlement_repo': entitlement_repo})
api.add_resource(SynthesisResource, '/datasets/<dataset_id>/synthesis', resource_class_kwargs={'dataset_repo': dataset_repo, 'synthesis_repo': synthesis_repo, 'synthesizer_manager': synthesizer_manager, 'entitlement_repo': entitlement_repo})
api.add_resource(SynthesisPreviewResource, '/datasets/<dataset_id>/synthesis-preview', resource_class_kwargs={'dataset_repo': dataset_repo, 'synthesizer_manager': synthesizer_manager, 'entitlement_repo': entitlement_repo})
api.add_resource(ReportResource, '/datasets/<dataset_id>/report', resource_class_kwargs={'dataset_repo': dataset_repo, 'report_repo': report_repo, 'entitlement_repo': entitlement_repo})
api.add_resource(DatasetExportResource, '/datasets/<dataset_id>/export', resource_class_kwargs={'dataset_repo': dataset_repo, 'synthesis_repo': synthesis_repo, 'entitlement_repo': entitlement_repo})
api.add_resource(ReportItemsResource, '/datasets/<dataset_id>/report-items', resource_class_kwargs={'dataset_repo': dataset_repo, 'report_repo': report_repo, 'report_item_repo': report_item_repo, 'entitlement_repo': entitlement_repo})
api.add_resource(ReportItemResource, '/datasets/<dataset_id>/report-items/<report_item_id>', resource_class_kwargs={'dataset_repo': dataset_repo, 'report_repo': report_repo, 'report_item_repo': report_item_repo, 'entitlement_repo': entitlement_repo})
api.add_resource(ReportItemsUpdateSettingsResource, '/datasets/<dataset_id>/report-items/<report_item_id>/updatesettings', resource_class_kwargs={'dataset_repo': dataset_repo, 'report_repo': report_repo, 'report_item_repo': report_item_repo, 'synthesis_repo': synthesis_repo, 'entitlement_repo': entitlement_repo})
api.add_resource(ReportItemsMoveResource, '/datasets/<dataset_id>/report-items/<report_item_id>/move', resource_class_kwargs={'dataset_repo': dataset_repo, 'report_repo': report_repo, 'report_item_repo': report_item_repo, 'report_item_ordering': report_item_ordering, 'entitlement_repo': entitlement_repo})
api.add_resource(ProjectTemplatesResource, '/templates', resource_class_kwargs={'template_directory': template_directory})
api.add_resource(DatasetFromTemplateResource, '/templates/<template_id>/dataset', resource_class_kwargs={'project_templates': project_templates})
api.add_resource(EntitlementsResource, '/datasets/<dataset_id>/entitlements', resource_class_kwargs={'dataset_repo': dataset_repo, 'user_repo': user_repo, 'entitlement_repo': entitlement_repo})
api.add_resource(EntitlementResource, '/datasets/<dataset_id>/entitlements/<entitlement_id>', resource_class_kwargs={'dataset_repo': dataset_repo, 'user_repo': user_repo, 'entitlement_repo': entitlement_repo})
api.add_resource(EntitlementCompletionResource, '/datasets/<dataset_id>/entitlements/complete', resource_class_kwargs={'dataset_repo': dataset_repo, 'entitlement_completion': entitlement_completion, 'entitlement_repo': entitlement_repo})
