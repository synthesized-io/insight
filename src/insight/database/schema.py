from sqlalchemy import FLOAT, INTEGER, TIMESTAMP, VARCHAR, Column, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Dataset(Base):
    __tablename__ = "dataset"

    id = Column(INTEGER, primary_key=True)
    name = Column(VARCHAR(50), nullable=False)
    num_rows = Column(INTEGER)
    num_columns = Column(INTEGER)
    created_at = Column(TIMESTAMP, default=func.now())


class Metric(Base):
    __tablename__ = "metric"

    id = Column(INTEGER, primary_key=True)
    name = Column(VARCHAR(50), nullable=False)
    category = Column(VARCHAR(50))
    created_at = Column(TIMESTAMP, default=func.now())


class Version(Base):
    __tablename__ = "version"

    id = Column(INTEGER, primary_key=True)
    name = Column(VARCHAR(50), nullable=False, default="unversioned")
    created_at = Column(TIMESTAMP, default=func.now())


class Result(Base):
    __tablename__ = "result"

    id = Column(INTEGER, primary_key=True)
    metric_id = Column(INTEGER, ForeignKey("metric.id"))
    dataset_id = Column(INTEGER, ForeignKey("dataset.id"))
    version_id = Column(INTEGER, ForeignKey("version.id"))
    value = Column(FLOAT)
    created_at = Column(TIMESTAMP, default=func.now())

    metric: Metric = relationship("Metric")
    dataset: Dataset = relationship("Dataset")
    version: Version = relationship("Version")
