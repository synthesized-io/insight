import os


class Config(object):
    SEND_FILE_MAX_AGE_DEFAULT = 600  # TODO: implement proper static files serving
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = False
    PROPAGATE_EXCEPTIONS = False  # to enforce json response
    JWT_SECRET_KEY = 'a4ebd40d'
    JWT_ERROR_MESSAGE_KEY = 'message'
    INVITE_KEY = 'be58c543'


class ProductionConfig(Config):
    SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}'.format(
        user=os.environ.get('RDS_USERNAME', None),
        pw=os.environ.get('RDS_PASSWORD', None),
        host=os.environ.get('RDS_HOSTNAME', None),
        port=os.environ.get('RDS_PORT', None),
        db=os.environ.get('RDS_DB_NAME', None))


class DevelopmentConfig(Config):
    SQLALCHEMY_DATABASE_URI = 'sqlite:////tmp/synthesized_web.db'
