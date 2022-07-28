import os

from sqlalchemy import FLOAT, INTEGER, JSON, TIMESTAMP, VARCHAR, Column, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class Dataset(Base):
    __tablename__ = "dataset"

    id = Column(INTEGER, primary_key=True)
    name = Column(VARCHAR(50), nullable=False)
    num_rows = Column(INTEGER)
    num_columns = Column(INTEGER)
    df_meta = Column(JSON)
    created_at = Column(TIMESTAMP)


class Metric(Base):
    __tablename__ = "metric"

    id = Column(INTEGER, primary_key=True)
    name = Column(VARCHAR(50), nullable=False)
    category = Column(VARCHAR(50))
    created_at = Column(TIMESTAMP)


class Version(Base):
    __tablename__ = "version"

    id = Column(INTEGER, primary_key=True)
    name = Column(VARCHAR(50), nullable=True)
    created_at = Column(TIMESTAMP)


class Result(Base):
    __tablename__ = "result"

    id = Column(INTEGER, primary_key=True)
    metric_id = Column(INTEGER, ForeignKey("metric.id"))
    dataset_id = Column(INTEGER, ForeignKey("dataset.id"))
    version_id = Column(INTEGER, ForeignKey("version.id"))
    value = Column(FLOAT)
    created_at = Column(TIMESTAMP)

    metric: Metric = relationship("Metric")
    dataset: Dataset = relationship("Dataset")
    version: Version = relationship("Version")


engine = create_engine(
        "postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@postgres:5432".format(**os.environ),
        future=True
        )
Session = sessionmaker(bind=engine, future=True)
