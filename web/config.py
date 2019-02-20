import os


class Config(object):
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class ProductionConfig(Config):
    SQLALCHEMY_DATABASE_URI = 'postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}'.format(
        user=os.environ['RDS_USERNAME'],
        pw=os.environ['RDS_PASSWORD'],
        host=os.environ['RDS_HOSTNAME'],
        port=os.environ['RDS_PORT'],
        db=os.environ['RDS_DB_NAME'])


class DevelopmentConfig(Config):
    SQLALCHEMY_DATABASE_URI = 'sqlite:////tmp/synthesized_web.db'