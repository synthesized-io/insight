from collections import namedtuple
from datetime import datetime
from enum import Enum
from io import BytesIO

import simplejson

from .dataset_meta import DatasetMeta
from ..app import db


class AuditMixin(object):
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)


class User(db.Model, AuditMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.Text, nullable=False, unique=True)
    password = db.Column(db.Text, nullable=False)
    first_name = db.Column(db.Text)
    last_name = db.Column(db.Text)
    phone_number = db.Column(db.Text)
    job_title = db.Column(db.Text)
    company = db.Column(db.Text)

    def __str__(self):
        return "<User {}>".format(self.id)


class UsedInvite(db.Model, AuditMixin):
    code = db.Column(db.Text, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey(User.id), nullable=False)

    def __str__(self):
        return "<UsedInvite {}>".format(self.code)


class Dataset(db.Model, AuditMixin):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey(User.id), nullable=False)
    title = db.Column(db.Text)
    description = db.Column(db.Text)
    blob = db.Column(db.LargeBinary, nullable=False)
    meta = db.Column(db.LargeBinary, nullable=False)
    syntheses = db.relationship("Synthesis", cascade="all, delete-orphan", lazy='select')
    reports = db.relationship("Report", cascade="all, delete-orphan", lazy='select')

    def get_meta_as_object(self) -> DatasetMeta:
        # Parse JSON into an object with attributes corresponding to dict keys.
        return simplejson.load(BytesIO(self.meta), encoding='utf-8', object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

    def set_meta_from_object(self, meta: DatasetMeta):
        self.meta = simplejson.dumps(meta, default=lambda x: x.__dict__, ignore_nan=True).encode('utf-8')

    def __str__(self):
        return '<Dataset {}>'.format(self.id)


class Synthesis(db.Model, AuditMixin):
    id = db.Column(db.Integer, primary_key=True)
    blob = db.Column(db.LargeBinary, nullable=False)
    meta = db.Column(db.LargeBinary, nullable=False)
    size = db.Column(db.Integer, nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey(Dataset.id), nullable=False)

    def get_meta_as_object(self) -> DatasetMeta:
        # Parse JSON into an object with attributes corresponding to dict keys.
        return simplejson.load(BytesIO(self.meta), encoding='utf-8', object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

    def set_meta_from_object(self, meta: DatasetMeta):
        self.meta = simplejson.dumps(meta, default=lambda x: x.__dict__, ignore_nan=True).encode('utf-8')

    def __str__(self):
        return '<Synthesis {}>'.format(self.id)


class Report(db.Model, AuditMixin):
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey(Dataset.id), nullable=False)
    items = db.relationship("ReportItem", cascade="all, delete-orphan", lazy='select')

    def __str__(self):
        return '<Report {}>'.format(self.id)


class ReportItemType(Enum):
    CORRELATION = 1
    MODELLING = 2


class ReportItem(db.Model, AuditMixin):
    id = db.Column(db.Integer, primary_key=True)
    ord = db.Column(db.Integer, nullable=False)
    report_id = db.Column(db.Integer, db.ForeignKey(Report.id), nullable=False)
    item_type = db.Column(db.Enum(ReportItemType), nullable=False)
    settings = db.Column(db.LargeBinary)
    results = db.Column(db.LargeBinary)

    def __str__(self):
        return '<ReportItem {}>'.format(self.id)


# this is not stored in the db
class ProjectTemplate:
    def __init__(self, id: int, title: str, description: str, file: str):
        self.id = id
        self.title = title
        self.description = description
        self.file = file

    def __str__(self):
        return '<Report {}>'.format(self.id)
