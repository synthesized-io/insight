from sqlalchemy import FLOAT, INTEGER, TIMESTAMP, VARCHAR, Column, ForeignKey
from sqlalchemy.orm import Mapped, declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()
mapped_column = Column

class Dataset(Base):
    __tablename__ = "dataset"

    id: Mapped[int] = mapped_column(INTEGER, primary_key=True)
    name = mapped_column(VARCHAR(200), nullable=False)
    num_rows = mapped_column(INTEGER)
    num_columns = mapped_column(INTEGER)
    created_at = mapped_column(TIMESTAMP, default=func.now())


class Metric(Base):
    __tablename__ = "metric"

    id = mapped_column(INTEGER, primary_key=True)
    name = mapped_column(VARCHAR(100), nullable=False)
    category = mapped_column(VARCHAR(100))
    created_at = mapped_column(TIMESTAMP, default=func.now())


class Version(Base):
    __tablename__ = "version"

    id = mapped_column(INTEGER, primary_key=True)
    name = mapped_column(VARCHAR(50), nullable=False, default="unversioned")
    created_at = mapped_column(TIMESTAMP, default=func.now())


class Result(Base):
    __tablename__ = "result"

    id = mapped_column(INTEGER, primary_key=True)
    metric_id = mapped_column(INTEGER, ForeignKey("metric.id"))
    dataset_id = mapped_column(INTEGER, ForeignKey("dataset.id"))
    version_id = mapped_column(INTEGER, ForeignKey("version.id"))
    value = mapped_column(FLOAT)
    created_at = mapped_column(TIMESTAMP, default=func.now())
    run_id = mapped_column(VARCHAR(50), nullable=True, default=None)

    metric: Mapped[Metric] = relationship("Metric")
    dataset: Mapped[Dataset] = relationship("Dataset")
    version: Mapped[Version] = relationship("Version")
