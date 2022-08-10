import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

db_url = "postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@"\
         "{POSTGRES_HOST}:{POSTGRES_PORT}".format(**os.environ)
engine = create_engine(db_url, future=True)

Session = sessionmaker(bind=engine, future=True)
