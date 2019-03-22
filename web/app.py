
from .middleware import configure_logger
configure_logger()

from flask import Flask
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_restful import Api
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager

from .config import ProductionConfig, DevelopmentConfig
from .middleware import JSONCompliantEncoder

from .repository import SQLAlchemyRepository
from .synthesizer_manager import SynthesizerManager
from .authenticator import Authenticator


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

jwt = JWTManager(app)

dataset_repo = SQLAlchemyRepository(db, Dataset)
synthesis_repo = SQLAlchemyRepository(db, Synthesis)
user_repo = SQLAlchemyRepository(db, User)
authenticator = Authenticator(user_repo, bcrypt)

# each model is about 275MB in RAM
synthesizer_manager = SynthesizerManager(dataset_repo=dataset_repo, max_models=15)

from .resources.status import StatusResource
from .resources.auth import LoginResource, RefreshResource, UsersResource
from .resources.dataset import DatasetsResource, DatasetResource, DatasetUpdateInfoResource
from .resources.synthesis import ModelResource, SynthesisResource

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
