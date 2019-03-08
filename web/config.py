import os


class Config(object):
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False
    PROPAGATE_EXCEPTIONS = True  # this is crucial to enable handling of JWTError
    JWT_SECRET_KEY = 'synthesized-secret'
    JWT_ERROR_MESSAGE_KEY = 'message'


class ProductionConfig(Config):
    SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}'.format(
        user=os.environ.get('RDS_USERNAME', None),
        pw=os.environ.get('RDS_PASSWORD', None),
        host=os.environ.get('RDS_HOSTNAME', None),
        port=os.environ.get('RDS_PORT', None),
        db=os.environ.get('RDS_DB_NAME', None))


class DevelopmentConfig(Config):
    SQLALCHEMY_DATABASE_URI = 'sqlite:////tmp/synthesized_web.db'
