from web.infastructure.flask_support import configure_logger
configure_logger()

from flask import Flask
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager

from .config import ProductionConfig, DevelopmentConfig
from .infastructure.flask_support import JSONCompliantEncoder
from .infastructure.repository_impl import SQLAlchemyRepository
from .application.synthesizer_manager import SynthesizerManager
from .application.authenticator import Authenticator


app = Flask(__name__)
if app.config['ENV'] == 'production':
    app.config.from_object(ProductionConfig)
else:
    app.config.from_object(DevelopmentConfig)
app.json_encoder = JSONCompliantEncoder

CORS(app, supports_credentials=True)  # TODO: delete in final version

bcrypt = Bcrypt(app)

jwt = JWTManager(app)

db = SQLAlchemy(app)
# models use `db` object, therefore should be imported after `db` creation
from web.domain.model import Dataset, Synthesis, User, Report, ReportItem
db.create_all()

dataset_repo = SQLAlchemyRepository(db, Dataset)
synthesis_repo = SQLAlchemyRepository(db, Synthesis)
user_repo = SQLAlchemyRepository(db, User)
report_repo = SQLAlchemyRepository(db, Report)
report_item_repo = SQLAlchemyRepository(db, ReportItem)


authenticator = Authenticator(user_repo, bcrypt)

# each model is about 275MB in RAM
synthesizer_manager = SynthesizerManager(dataset_repo=dataset_repo, max_models=15)

# should be imported after `db` creation
from .infastructure.report_item_ordering_imp import SQLAlchemyReportItemOrdering
report_item_ordering = SQLAlchemyReportItemOrdering(db)

from .resources.status import StatusResource
from .resources.auth import LoginResource, RefreshResource, UsersResource
from .resources.dataset import DatasetsResource, DatasetResource, DatasetUpdateInfoResource
from .resources.synthesis import ModelResource, SynthesisResource
from .resources.report import ReportItemsResource, ReportResource, ReportItemsUpdateSettingsResource, ReportItemsMoveResource, ReportItemResource

api = Api(app)
api.add_resource(StatusResource, '/')
api.add_resource(LoginResource, '/login', resource_class_kwargs={'authenticator': authenticator})
api.add_resource(RefreshResource, '/refresh')
api.add_resource(UsersResource, '/users', resource_class_kwargs={'user_repo': user_repo, 'bcrypt': bcrypt})
api.add_resource(DatasetsResource, '/datasets', resource_class_kwargs={'dataset_repo': dataset_repo})
api.add_resource(DatasetResource, '/datasets/<dataset_id>', resource_class_kwargs={'dataset_repo': dataset_repo})
api.add_resource(DatasetUpdateInfoResource, '/datasets/<dataset_id>/updateinfo', resource_class_kwargs={'dataset_repo': dataset_repo})
api.add_resource(ModelResource, '/datasets/<dataset_id>/model', resource_class_kwargs={'dataset_repo': dataset_repo, 'synthesizer_manager': synthesizer_manager})
api.add_resource(SynthesisResource, '/datasets/<dataset_id>/synthesis', resource_class_kwargs={'dataset_repo': dataset_repo, 'synthesis_repo': synthesis_repo, 'synthesizer_manager': synthesizer_manager})
api.add_resource(ReportResource, '/datasets/<dataset_id>/report', resource_class_kwargs={'dataset_repo': dataset_repo, 'report_repo': report_repo})
api.add_resource(ReportItemsResource, '/datasets/<dataset_id>/report-items', resource_class_kwargs={'dataset_repo': dataset_repo, 'report_repo': report_repo, 'report_item_repo': report_item_repo})
api.add_resource(ReportItemResource, '/datasets/<dataset_id>/report-items/<report_item_id>', resource_class_kwargs={'dataset_repo': dataset_repo, 'report_repo': report_repo, 'report_item_repo': report_item_repo})
api.add_resource(ReportItemsUpdateSettingsResource, '/datasets/<dataset_id>/report-items/<report_item_id>/updatesettings', resource_class_kwargs={'dataset_repo': dataset_repo, 'report_repo': report_repo, 'report_item_repo': report_item_repo, 'synthesis_repo': synthesis_repo})
api.add_resource(ReportItemsMoveResource, '/datasets/<dataset_id>/report-items/<report_item_id>/move', resource_class_kwargs={'dataset_repo': dataset_repo, 'report_repo': report_repo, 'report_item_repo': report_item_repo, 'report_item_ordering': report_item_ordering})
