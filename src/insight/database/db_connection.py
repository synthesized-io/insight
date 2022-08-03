import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

db_url = "postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@postgres:5432".format(**os.environ)
engine = create_engine(db_url, future=True)

Session = sessionmaker(bind=engine, future=True)
